# Copyright 2022 Michael Hu. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import multiprocessing.sharedctypes
import multiprocessing.synchronize
from absl import logging
from typing import Callable, Iterable, List, Optional, Tuple, Mapping, Text, Any, TypeVar
from pathlib import Path
import sys
import signal
import copy
import pickle
import time
import queue
import multiprocessing
import gym
import logging

import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn.functional as F

from muzero.config import MuZeroConfig
from muzero.games.env import BoardGameEnv
from muzero.grokfast import gradfilter_ema
from muzero.network import MuZeroNet
from muzero.replay import Transition, PrioritizedReplay
from muzero.trackers import make_actor_trackers, make_learner_trackers, make_evaluator_trackers
from muzero.mcts import continous_annealing, uct_search
from muzero.util import scalar_to_categorical_probabilities, logits_to_transformed_expected_value
from muzero.rating import compute_elo_rating
from muzero.continous.io import ContinousActionDecoder
from muzero.continous.net import ContinousMuzeroNet
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


O = TypeVar('O')
A = TypeVar('A')


@torch.no_grad()
def run_self_play(
    config: MuZeroConfig,
    rank: int,
    network_config: dict[Any, Any] | MuZeroNet,
    device: torch.device,
    env: gym.Env[O, A],
    data_queue: multiprocessing.SimpleQueue,  # [tuple[Transition, float]],
    train_steps_counter: multiprocessing.Value,
    stop_event: multiprocessing.synchronize.Event,
    tag: Optional[str] = None,
    no_mask: bool = False,
    use_distance: Optional[Callable[[float, float], torch.Tensor]] = None,
    action_decoder: Optional[ContinousActionDecoder] = None,
    action_encoder: Optional[Callable[[int], torch.Tensor]] = None,
) -> None:
    """Run self-play for as long as needed, only stop if `stop_event` is set to True.

    Args:
        config: a MuZeroConfig instance.
        rank: actor process rank.
        network: a MuZeroNet instance for acting.
        device: PyTorch runtime device.
        env: actor's env.
        data_queue: a multiprocessing.Queue instance to send samples to leaner.
        train_steps_counter: a multiprocessing.Value instance to count current training steps.
        stop_event: a multiprocessing.Event instance signals stop run pipeline.
        tag: add tag to tensorboard log dir.
    """

    handle_exit_signal()
    logging.info(f'Start self-play actor {rank}')

    tb_log_dir = f'actor{rank}'
    if tag is not None and tag != '':
        tb_log_dir = f'{tag}_{tb_log_dir}'

    trackers = make_actor_trackers(tb_log_dir) if config.use_tensorboard else []
    for tracker in trackers:
        tracker.reset()

    if isinstance(network_config, MuZeroNet):
        network = network_config
    else:
        network = ContinousMuzeroNet(**network_config)

    network = network.to(device=device)

    logging.debug(f"actor device: {[p.device for p in network.parameters()]}")
    network.eval()
    game = 0

    while not stop_event.is_set():  # For each new game.
        obs = env.reset()
        logging.debug(f" ================= obs shape: {obs.shape}")
        done = False
        episode_trajectory = []
        steps = 0

        # Play and record transitions.
        # the second check is necessary becase the pipeline could have already stopped while the actor is in the middle of a game.
        while not done and not stop_event.is_set():
            # Make a copy of current player id.
            player_id = copy.deepcopy(env.current_player)
            action, pi_prob, root_value = uct_search(
                state=obs,
                network=network,
                device=device,
                config=config,
                temperature=config.visit_softmax_temperature_fn(steps, train_steps_counter.value),
                actions_mask=None if no_mask else env.actions_mask,
                current_player=env.current_player,
                opponent_player=env.opponent_player,
                action_decoder=action_decoder,
                action_encoder=action_encoder,
                step=train_steps_counter.value + steps,
                distance_projection=use_distance,
            )

            logging.info(f"predicted action: {action}")
            logging.debug(f"probabilistic choice: {pi_prob.argmax()}")

            next_obs, reward, done, _ = env.step(action)
            logging.debug(f" ================= next_obs shape: {next_obs.shape}")
            steps += 1
            action = action if action_encoder is None else action_encoder(action)
            pi_prob = pi_prob if action_encoder is None else action_encoder(pi_prob.argmax()).cpu().numpy()
            for tracker in trackers:
                tracker.step(reward, done, continous_annealing(train_steps_counter.value + steps))

            if not isinstance(action, torch.Tensor):
                logging.debug(f" ================= action: {action}")
                logging.debug(f" ================= type(action): {type(action)}")
                logging.debug(f" ================= step: {steps}")
            episode_trajectory.append((obs, action, reward, pi_prob, root_value, player_id))
            obs = next_obs

            # Send samples to learner every N steps on Atari games.
            # Here we accmulate N + unroll_steps + td_steps because
            # we needs these extra sequences to compute the target and unroll sequences.
            if (
                not config.is_board_game
                and len(episode_trajectory) == config.acc_seq_length + config.unroll_steps + config.td_steps
            ):
                logging.debug("send extra sequence")
                # Unpack list of tuples into seperate lists.
                observations, actions, rewards, pi_probs, root_values, _ = map(list, zip(*episode_trajectory))
                # Compute n_step target value.
                target_values = compute_n_step_target(rewards, root_values, config.td_steps, config.discount)

                priorities = np.abs(np.array(root_values) - np.array(target_values))

                # Make unroll sequences and send to learner.
                for transition, priority in make_unroll_sequence(
                    observations[: config.acc_seq_length],
                    actions[: config.acc_seq_length + config.unroll_steps],
                    rewards[: config.acc_seq_length + config.unroll_steps],
                    pi_probs[: config.acc_seq_length + config.unroll_steps],
                    target_values[: config.acc_seq_length + config.unroll_steps],
                    priorities[: config.acc_seq_length + config.unroll_steps],
                    config.unroll_steps,
                ):
                    # logging.debug(" transition: ", transition)
                    # logging.debug(" priority: ", priority)
                    data_queue.put((transition, priority))

                del episode_trajectory[: config.acc_seq_length]
                del (observations, actions, rewards, pi_probs, root_values, priorities, target_values)

        game += 1

        # Unpack list of tuples into seperate lists.
        observations, actions, rewards, pi_probs, root_values, player_ids = map(list, zip(*episode_trajectory))

        if config.is_board_game:
            # Using MC returns as target value.
            target_values = compute_mc_return_target(rewards, player_ids)
        else:
            # Compute n_step target value.
            target_values = compute_n_step_target(rewards, root_values, config.td_steps, config.discount)

        priorities = np.abs(np.array(root_values) - np.array(target_values))

        logging.debug(f" ================= len(episode_trajectory): {len(episode_trajectory)}")
        logging.debug(" ================= sending full sequence")
        # Make unroll sequences and send to learner.
        try:
            actions = [torch.tensor(a) for a in actions]
            for transition, priority in make_unroll_sequence(
                observations, actions, rewards, pi_probs, target_values, priorities, config.unroll_steps
            ):
                data_queue.put((transition, priority))
        except Exception as e:
            logging.error(f" ================= error: {e}")
            logging.error(f" ================= actions: {actions}")

        del episode_trajectory[:]
        del (observations, actions, rewards, pi_probs, root_values, priorities, player_ids, target_values)

    logging.info(f'Stop self-play actor {rank}')


def collect_grad_means(model: torch.nn.Module) -> list[Tuple[str, float]]:
    grads: list[Tuple[str, float]] = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None and "bias" not in name:
            mean_grad = param.grad.abs().mean().item()  # Get the mean of the absolute gradient
            grads.append((name, mean_grad))
    return grads


def run_training(  # noqa: C901
    config: MuZeroConfig,
    network: MuZeroNet,
    optimizer_state: dict[Any, Any],
    scheduler_state: dict[Any, Any],
    device: torch.device,
    actor_network: torch.nn.Module,
    replay: PrioritizedReplay,
    data_queue: multiprocessing.SimpleQueue,
    train_steps_counter: multiprocessing.Value,
    checkpoint_dir: str,
    checkpoint_files: List,
    stop_event: multiprocessing.Event,
    tag: str = None,
) -> None:
    """Run the main training loop for N iterations, each iteration contains M updates.
    This controls the 'pace' of the pipeline, including when should the other parties to stop.

    Args:
        config: a MuZeroConfig instance.
        network: the neural network we want to optimize.
        optimizer: neural network optimizer.
        lr_scheduler: learning rate annealing scheduler.
        device: torch runtime device.
        actor_network: the neural network actors runing self-play, for the case AlphaZero pipeline without evaluation.
        replay: a simple uniform experience replay.
        train_steps_counter: a multiprocessing.Value instance represent current training steps, shared with actors.
        checkpoint_dir: create new checkpoint save directory.
        checkpoint_files: a shared list contains the full path for the most recent new checkpoint files.
        stop_event: a multiprocessing.Event signaling other parties to stop running pipeline.
        tag: add tag to tensorboard log dir and checkpoint file name.

    """

    logging.info('Start training thread')

    tb_log_dir = 'learner'
    ckpt_prefix = 'train_steps'

    if tag is not None and tag != '':
        tb_log_dir = f'{tag}_{tb_log_dir}'
        ckpt_prefix = f'{tag}_{ckpt_prefix}'

    trackers = make_learner_trackers(tb_log_dir) if config.use_tensorboard else []
    for tracker in trackers:
        tracker.reset()

    network = network.to(device=device)
    network.train()
    # support legacy configurations
    if isinstance(optimizer_state, torch.optim.Adam):
        optimizer = optimizer_state
    else:
        optimizer = torch.optim.Adam(network.parameters(), lr=config.lr_init, weight_decay=config.weight_decay)
        if not (optimizer_state is None):
            optimizer.load_state_dict(optimizer_state)

    # lr_scheduler = MultiStepLR(optimizer, milestones=config.lr_milestones, gamma=config.lr_decay_rate)
    # support legacy configurations
    if (not scheduler_state is None) and not isinstance(scheduler_state, dict):
        lr_scheduler = scheduler_state
    else:
        lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5000, T_mult=1)
        if not (scheduler_state is None):
            lr_scheduler.load_state_dict(scheduler_state)
    # logging.debug("network device: ", network.device)

    ckpt_dir = Path(checkpoint_dir)
    if checkpoint_dir is not None and checkpoint_dir != '' and not ckpt_dir.exists():
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    def get_state_to_save():
        return {
            'network': network.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'train_steps': train_steps_counter.value,
        }

    grads = None
    last_log = time.time()
    while True:
        if replay.size < config.min_replay_size or replay.size < config.batch_size:
            if time.time() - last_log > 10:
                last_log = time.time()
                logging.info(f"replay size: {replay.size}")
            continue

        if train_steps_counter.value >= config.num_training_steps:
            break

        transitions, indices, weights = replay.sample(config.batch_size)
        logging.debug(f"Transitions:\n{transitions}")
        logging.debug(f"Indices:\n{indices}")
        logging.debug(f"Weights:\n{weights}")
        weights = torch.from_numpy(weights).to(device=device, dtype=torch.float32)

        optimizer.zero_grad()
        loss, priorities, individual_losses = calc_loss(network, device, transitions, weights)

        reward_loss, value_loss, policy_loss = individual_losses
        reward_loss = reward_loss.detach().cpu().item()
        value_loss = value_loss.detach().cpu().item()
        policy_loss = policy_loss.detach().cpu().item()

        loss.backward()


        # if config.clip_grad:
        #     torch.nn.utils.clip_grad_norm_(network.parameters(), config.max_grad_norm)
        # grokfast
        grads = gradfilter_ema(network, grads=grads, alpha=0.98, lamb=2.0)

        optimizer.step()
        lr_scheduler.step()

        if priorities is not None:
            if priorities.shape != (config.batch_size,):
                raise RuntimeError(f'Expect priorities has shape ({config.batch_size}, ), got {priorities.shape}')
            replay.update_priorities(indices, priorities)

        train_steps_counter.value += 1

        del transitions, indices, weights

        for tracker in trackers:
            tracker.step(
                loss.detach().cpu().item(),
                reward_loss,
                value_loss,
                policy_loss,
                lr_scheduler.get_last_lr()[0],
                train_steps_counter.value,
                network,
                continous_annealing(train_steps_counter.value),
            )

        if train_steps_counter.value > 1 and train_steps_counter.value % config.checkpoint_interval == 0:
            state_to_save = get_state_to_save()
            ckpt_file = ckpt_dir / f'{ckpt_prefix}_{train_steps_counter.value}'
            create_checkpoint(state_to_save, ckpt_file)
            checkpoint_files.append(ckpt_file)
            actor_network.load_state_dict(network.state_dict())
            actor_network.eval()

            del state_to_save

        # Wait for sometime before start training on next batch.
        if config.train_delay is not None and config.train_delay > 0 and train_steps_counter.value > 1:
            time.sleep(config.train_delay)

    # Signaling other parties to stop running pipeline.
    stop_event.set()
    time.sleep(10)
    data_queue.put('STOP')

    # Create a final checkpoint.
    state_to_save = get_state_to_save()
    ckpt_file = ckpt_dir / f'{ckpt_prefix}_{train_steps_counter.value}_final'
    create_checkpoint(state_to_save, ckpt_file)


def run_board_game_evaluator(
    config: MuZeroConfig,
    old_checkpoint_network: torch.nn.Module,
    new_ckpt_network: torch.nn.Module,
    device: torch.device,
    env: BoardGameEnv,
    temperature: float,
    checkpoint_files: List,
    stop_event: multiprocessing.Event,
    initial_elo: int = -2000,
    tag: str = None,
    action_decoder: Optional[Callable] = None,
) -> None:
    """
    Monitoring training progress by play a single game with new checkpoint againt last checkpoint.
    This is for board game only.

    Args:
        config: a MuZeroConfig instance.
        old_checkpoint_network: the last checkpoint network.
        new_ckpt_network: new checkpoint network we want to evaluate.
        device: torch runtime device.
        env: a BoardGameEnv type environment.
        temperature: the temperature exploration rate after MCTS search
            to generate play policy.
        checkpoint_files: a shared list contains the full path for the most recent new checkpoint.
        stop_event: a multiprocessing.Event signaling to stop running pipeline.
        initial_elo: initial elo ratings for the players, default -2000.
        tag: add tag to tensorboard log dir.

     Raises:
        ValueError:
            if `env` is not a valid BoardGameEnv instance.
    """
    if not isinstance(env, BoardGameEnv):
        raise ValueError(f'Expect env to be a valid BoardGameEnv instance, got {env}')

    handle_exit_signal()
    logging.info('Start evaluator')

    tb_log_dir = 'evaluator'

    if tag is not None and tag != '':
        tb_log_dir = f'{tag}_{tb_log_dir}'

    trackers = make_evaluator_trackers(tb_log_dir, True) if config.use_tensorboard else []
    for tracker in trackers:
        tracker.reset()

    disable_auto_grad(old_checkpoint_network)
    disable_auto_grad(new_ckpt_network)

    old_checkpoint_network = old_checkpoint_network.to(device=device)
    new_ckpt_network = new_ckpt_network.to(device=device)

    # Set initial elo ratings
    black_elo = initial_elo
    white_elo = initial_elo

    while True:
        if stop_event.is_set() and len(checkpoint_files) == 0:
            break
        if len(checkpoint_files) == 0:
            continue

        # Remove the checkpoint file path from the shared list.
        ckpt_file = checkpoint_files.pop(0)
        loaded_state = load_checkpoint(ckpt_file, device)
        new_ckpt_network.load_state_dict(loaded_state['network'])
        train_steps = loaded_state['train_steps']
        del loaded_state

        new_ckpt_network.eval()
        old_checkpoint_network.eval()

        obs = env.reset()
        done = False

        while not done:
            # Black is the new checkpoint, white is last checkpoint.
            if env.current_player == env.black_player_id:
                network = new_ckpt_network
            else:
                network = old_checkpoint_network

            action, *_ = uct_search(
                state=obs,
                network=network,
                device=device,
                config=config,
                temperature=temperature,
                actions_mask=env.actions_mask,
                current_player=env.current_player,
                opponent_player=env.opponent_player,
                deterministic=True,
            )
            if action_decoder is not None:
                action = action_decoder(action)

            obs, _, done, _ = env.step(action)

        if env.winner == env.black_player_id:
            black_elo, _ = compute_elo_rating(0, black_elo, white_elo)
        elif env.winner == env.white_player_id:
            black_elo, _ = compute_elo_rating(1, black_elo, white_elo)
        white_elo = black_elo

        for tracker in trackers:
            tracker.step(black_elo, env.steps, train_steps)

        old_checkpoint_network.load_state_dict(new_ckpt_network.state_dict())


def run_evaluator(
    config: MuZeroConfig,
    new_ckpt_network: torch.nn.Module,
    device: torch.device,
    env: BoardGameEnv,
    temperature: float,
    checkpoint_files: List,
    stop_event: multiprocessing.Event,
    tag: str = None,
    num_episodes: int = 1,
    no_mask: bool = False,
    distance_projection: Optional[Callable[[float, float], torch.Tensor]] = None,
    action_decoder: Optional[Callable] = None,
    action_encoder: Optional[Callable] = None,
) -> None:
    """
    Monitoring training progress by play few games with the most recent new checkpoint.

    Args:
        config: a MuZeroConfig instance.
        new_ckpt_network: new checkpoint network we want to evaluate.
        device: torch runtime device.
        env: a BoardGameEnv type environment.
        temperature: the temperature exploration rate after MCTS search
            to generate play policy.
        checkpoint_files: a shared list contains the full path for the most recent new checkpoint.
        stop_event: a multiprocessing.Event signaling to stop running pipeline.
        tag: add tag to tensorboard log dir.
        num_episodes: run evaluation for N episodes, default 1.

    """

    handle_exit_signal()
    logging.info('Start evaluator')

    tb_log_dir = 'evaluator'

    if tag is not None and tag != '':
        tb_log_dir = f'{tag}_{tb_log_dir}'

    trackers = make_evaluator_trackers(tb_log_dir, False) if config.use_tensorboard else []
    for tracker in trackers:
        tracker.reset()

    disable_auto_grad(new_ckpt_network)
    new_ckpt_network = new_ckpt_network.to(device=device)

    while True:
        if stop_event.is_set() and len(checkpoint_files) == 0:
            break
        if len(checkpoint_files) == 0:
            continue

        # Remove the checkpoint file path from the shared list.
        ckpt_file = checkpoint_files.pop(0)
        loaded_state = load_checkpoint(ckpt_file, device)
        new_ckpt_network.load_state_dict(loaded_state['network'])
        train_steps = loaded_state['train_steps']
        del loaded_state

        new_ckpt_network.eval()

        eval_returns, eval_steps = [], []

        for _ in range(num_episodes):
            obs = env.reset()
            done = False
            steps = 0
            returns = 0.0

            while not done:
                action, *_ = uct_search(
                    state=obs,
                    network=new_ckpt_network,
                    device=device,
                    config=config,
                    temperature=temperature,
                    actions_mask=None if no_mask else env.actions_mask,
                    current_player=env.current_player,
                    opponent_player=env.opponent_player,
                    deterministic=True,
                    action_encoder=action_encoder,
                    action_decoder=action_decoder,
                    distance_projection=distance_projection,
                )

                action = action if action_decoder is not None else action_decoder(action)

                obs, reward, done, _ = env.step(action)
                steps += 1
                returns += reward

            eval_returns.append(returns)
            eval_steps.append(steps)

        for tracker in trackers:
            tracker.step(eval_returns, eval_steps, train_steps)


def run_data_collector(
    data_queue: multiprocessing.SimpleQueue,  # [tuple[Transition, float] | str],
    replay: PrioritizedReplay[Transition],
    save_frequency: int,
    save_dir: Optional[str],
    tag: Optional[str] = None,
) -> None:
    """Collect samples from self-play,
    this runs on the same process as the training loop,
    but with a separate thread.

    Args:
        data_queue: a multiprocessing.SimpleQueue to receive samples from self-play processes.
        replay: a simple uniform random experience replay.
        save_frequency: the frequency to save replay state.
        save_dir: where to save replay state.
        stop_event: a multiprocessing.Event signaling stop runing pipeline.

    """

    logging.info('Start data collector thread')

    samples_prefix = 'replay'

    if tag is not None and tag != '':
        samples_prefix = f'{tag}_{samples_prefix}'

    if save_dir is None:
        raise ValueError("save_dir not provided")
    save_samples_dir = Path(save_dir)
    if save_dir != '' and not save_samples_dir.exists():
        save_samples_dir.mkdir(parents=True, exist_ok=True)

    should_save = save_samples_dir.exists() and save_frequency > 0

    while True:
        try:
            item = data_queue.get()
            if item == 'STOP':
                break
            if isinstance(item, str):
                continue
            transition, priority = item
            replay.add(transition, priority)
            if should_save and replay.num_added > 1 and replay.num_added % save_frequency == 0:
                save_file = save_samples_dir / f'{samples_prefix}_{replay.size}_{get_time_stamp(True)}'
                save_to_file(replay.get_state(), str(save_file))
                logging.info(f"Replay samples saved to '{save_file}'")
        except queue.Empty:
            pass
        except EOFError:
            pass


def calc_loss(
    network: MuZeroNet,
    device: torch.device,
    transitions: Transition,
    weights: torch.Tensor,
) -> Tuple[torch.Tensor, np.ndarray, tuple[torch.Tensor]]:
    """Given a network and batch of transitions, compute the loss for MuZero agent."""
    # [B, state_shape]
    state = torch.from_numpy(transitions.state).to(device=device, dtype=torch.float32, non_blocking=True)
    # [B, T]
    action = torch.from_numpy(transitions.action).to(device=device, dtype=torch.long, non_blocking=True)
    target_value_scalar = torch.from_numpy(transitions.value).to(device=device, dtype=torch.float32, non_blocking=True)
    target_reward_scalar = torch.from_numpy(transitions.reward).to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    # [B, T, num_actions]
    target_pi_prob = torch.from_numpy(transitions.pi_prob).to(device=device, dtype=torch.float32, non_blocking=True)

    # Convert scalar targets into transformed support (probabilities).
    if network.mse_loss_for_value:
        target_value = target_value_scalar
    else:
        # [B, T, num_actions]
        target_value = scalar_to_categorical_probabilities(target_value_scalar, network.value_support_size)
    if network.mse_loss_for_reward:
        target_reward = target_reward_scalar
    else:
        # [B, T, num_actions]
        target_reward = scalar_to_categorical_probabilities(target_reward_scalar, network.reward_support_size)

    B, T = action.shape
    reward_loss, value_loss, policy_loss = (0, 0, 0)
    loss_scale = 1.0 / T

    pred_values = []

    # Get initial hidden state.
    hidden_state = network.represent(state)

    # Unroll K steps.
    for t in range(T):
        pred_pi_logits, pred_value = network.prediction(hidden_state)
        hidden_state, pred_reward = network.dynamics(hidden_state, action[:, t].unsqueeze(1))

        # Scale the gradient for dynamics function by 0.5.
        hidden_state.register_hook(lambda grad: grad * 0.5)

        value_loss += loss_func(pred_value.squeeze(), target_value[:, t], network.mse_loss_for_value)

        # Shouldn't we exclude the reward loss for board games as stated in the paper?
        reward_loss += loss_func(pred_reward.squeeze(), target_reward[:, t], network.mse_loss_for_reward)
        policy_loss += loss_func(pred_pi_logits, target_pi_prob[:, t])

        pred_values.append(pred_value.detach())

    loss = reward_loss + value_loss + policy_loss

    # Scale loss using importance sampling weights, then average over batch dimension B.
    loss = torch.mean(loss * weights.detach())

    # Scale the loss by 1/unroll_steps.
    loss.register_hook(lambda grad: grad * loss_scale)

    # Using the delta between predicted values and target values as priorities.
    with torch.no_grad():
        pred_values = torch.stack(pred_values, dim=1)
        if network.mse_loss_for_value:
            pred_values_scalar = pred_values.squeeze(-1)
        else:
            pred_values_scalar = logits_to_transformed_expected_value(pred_values, network.value_support_size).squeeze(
                -1
            )
        priorities = (pred_values_scalar[:, 0] - target_value_scalar[:, 0]).abs().cpu().numpy()
        # priorities = torch.mean(pred_values_scalar - target_value_scalar, dim=-1).abs().cpu().numpy()

    return loss, priorities, (reward_loss, value_loss, policy_loss)


def _calc_loss(
    network: MuZeroNet,
    device: torch.device,
    transitions: Transition,
    weights: torch.Tensor,
) -> Tuple[torch.Tensor, np.ndarray]:
    """Given a network and batch of transitions, compute the loss for MuZero agent."""
    # [B, state_shape]
    state = torch.from_numpy(transitions.state).to(device=device, dtype=torch.float32, non_blocking=True)
    # logging.debug(f"state.shape: {state.shape}")
    # [B, T]
    action = torch.from_numpy(transitions.action).to(device=device, dtype=torch.long, non_blocking=True)

    # logging.debug("action shape in loss: ", action.shape)
    target_value_scalar = (
        torch.from_numpy(transitions.value).to(device=device, dtype=torch.float32, non_blocking=True).unsqueeze(-1)
    )

    # logging.debug("target_value_scalar shape: ", target_value_scalar.shape)
    target_reward_scalar = (
        torch.from_numpy(transitions.reward).to(device=device, dtype=torch.float32, non_blocking=True).unsqueeze(-1)
    )

    # logging.debug("target_reward_scalar shape: ", target_reward_scalar.shape)
    # [B, T, num_actions]
    # the vector that was predicted by the policy head of the network
    target_pi_prob = torch.from_numpy(transitions.pi_prob)

    # assert torch.cdist(network.action_decoder(target_pi_prob.to(device=device)), target_pi_prob).sum() == 0.00001 # check if the action decoder is working correctly

    # selected_action_idx = network.action_decoder.index(target_pi_prob)
    target_pi_prob = target_pi_prob.to(device=device, dtype=torch.float32, non_blocking=True)
    # logging.debug("selected_action_idx: ", selected_action_idx)
    # get every other action in network.action_decoder.action_set
    # selected_action_idx has size [B, T]
    #     non_targets = []
    # #    logging.debug("not selected", no_select)
    #     for B in range(target_pi_prob.shape[0]):
    #         for i in range(target_pi_prob.shape[1]):
    #             # logging.debug(f"Best match for {network.action_decoder.action_set[B,i]} is {selected_action_idx[B][i]}")
    #             not_selected = network.action_decoder.action_set[~(network.action_decoder.action_set == selected_action_idx[B][i]).all(dim=-1)]
    #             non_targets.append(not_selected)

    # logging.debug("non_targets_shape")

    # logging.debug("target_pi_prob shape: ", target_pi_prob.shape)
    # Convert scalar targets into transformed support (probabilities).
    # if network.mse_loss_for_value:
    #     target_value = target_value_scalar
    # else:
    #     # [B, T, num_actions]
    target_value = scalar_to_categorical_probabilities(target_value_scalar.squeeze(-1), network.value_support_size)
    logging.debug(f"target_value categorical shape: {target_value.shape}")

    # if network.mse_loss_for_reward:
    #     target_reward = target_reward_scalar
    # else:
    #     # [B, T, num_actions]
    target_reward = scalar_to_categorical_probabilities(target_reward_scalar.squeeze(-1), network.reward_support_size)
    logging.debug(f"target_reward categorical shape: {target_reward.shape}")

    B, T, V = action.shape
    # logging.debug("B: ", B)
    # logging.debug("T: ", T)
    # logging.debug("V: ", V)
    reward_loss, value_loss, policy_loss = (0, 0, 0)
    loss_scale = 1.0  # / T

    # target_value = target_value_scalar.unsqueeze(-1).expand(B, T)
    # target_reward = target_reward_scalar.unsqueeze(-1).expand(B, T)
    # lengths = torch.sqrt(torch.sum(action ** 2, axis=-1))

    # Calculate the regularization term
    # regularization_term = torch.sum((lengths - 1) ** 2)

    # # Define the regularization coefficient
    # lambda_coefficient = 0.1  # This is a hyperparameter that you can tune

    pred_values = []

    # Get initial hidden state.
    hidden_state = network.represent(state)
    directional_sensitivity = 10.0

    # Unroll K steps.
    for t in range(T):
        # logging.debug(f"mag hidden_state: {torch.sqrt(torch.sum(hidden_state * hidden_state, dim=-1))}")
        pred_pi_logits, pred_value = network.prediction(hidden_state)
        # logging.debug(f"predicted value [{t}]: {pred_value}")
        # logging.debug("pred_value shape: ", pred_value.shape)
        # logging.debug("pred_pi_logits shape: ", pred_pi_logits.shape)
        # logging.debug("pred_value shape: ", pred_value.shape)
        action_slice = action[:, t, :]
        # logging.debug("action slice shape: ", action_slice.shape)
        hidden_state, pred_reward = network.dynamics(hidden_state, action_slice)

        # logging.debug("=====================================")
        # Scale the gradient for dynamics function by 0.5.
        # hidden_state.register_hook(lambda grad: grad * 0.5)
        # logging.debug("pred_value shape: ", pred_reward.shape)
        # logging.debug("target_value shape: ", target_value.shape)
        # logging.debug(" Value net: ")
        t_v = target_value[:, t]
        # logging.debug("t_v shape: ", t_v.shape)
        # logging.debug("t_v: ", t_v)
        value_loss += loss_func(pred_value.squeeze(), t_v, mse=True, cosine=False) * T
        # logging.debug("value_loss: ", value_loss)

        # Shouldn't we exclude the reward loss for board games as stated in the paper?
        # logging.debug("pred_reward shape: ", pred_reward.shape)
        # logging.debug("target_reward shape: ", target_reward.shape)
        # logging.debug(" Reward net: ")
        t_r = target_reward[:, t]
        # logging.debug("t_r shape: ", t_r.shape)
        # logging.debug("t_r: ", t_r)
        reward_loss += loss_func(pred_reward.squeeze(), t_r, mse=True, cosine=False) * T
        # logging.debug("reward_loss: ", reward_loss)

        # logging.debug("pred_pi_logits shape: ", pred_pi_logits.shape)
        # logging.debug("target_pi_prob shape: ", target_pi_prob.shape)
        # logging.debug(" Policy net: ")
        # TODO: in case of continous~categorical (classical control)
        #       we need a way to steer output intensity (maybe vector magnitude)
        policy_loss += loss_func(pred_pi_logits, target_pi_prob[:, t])
        # policy_loss += margin_cosine_embedding_loss(pred_pi_logits, target_pi_prob[:, t], non_targets=non_targets)
        # logging.debug("policy_loss: ", policy_loss)
        # logging.debug("=====================================")

        pred_values.append(pred_value.detach())

    # logging.debug("reward_loss sum(): ", reward_loss.sum())
    # logging.debug("value_loss sum(): ", value_loss.sum())
    # logging.debug("policy_loss sum(): ", policy_loss.sum())

    loss = reward_loss + value_loss + policy_loss  # + lambda_coefficient * regularization_term
    # logging.debug("loss shape: ", loss.shape)
    # logging.debug("weights shape: ", weights.shape)

    # Scale loss using importance sampling weights, then average over batch dimension B.
    loss = torch.mean(loss.to(device) * weights.detach().to(device).unsqueeze(-1))
    # logging.debug("loss: ", loss)

    # Scale the loss by 1/unroll_steps.
    # loss.register_hook(lambda grad: grad * loss_scale)

    # Using the delta between predicted values and target values as priorities.
    with torch.no_grad():
        pred_values = torch.stack(pred_values, dim=1)
        # logging.debug("pred_values shape: ", pred_values.shape)
        # logging.debug("network.mse_loss_for_value: ", network.mse_loss_for_value)
        # if network.mse_loss_for_value:
        #     pred_values_scalar = pred_values.squeeze(-1)
        # else:
        pred_values_scalar = logits_to_transformed_expected_value(pred_values, network.value_support_size).squeeze(-1)

        # logging.debug("pred_values_scalar shape: ", pred_values_scalar.shape)
        # logging.debug("target_value shape: ", target_value.shape)
        priorities = (pred_values_scalar - target_value_scalar.squeeze(-1)).abs().cpu().numpy()
        priorities = torch.mean(pred_values_scalar - target_value_scalar.squeeze(-1), dim=-1).abs().cpu().numpy()
        # logging.debug("priorities: ", priorities)

    return loss, priorities


def margin_cosine_embedding_loss(
    prediction: torch.Tensor, target: torch.Tensor, non_targets: torch.Tensor, margin: float = 0.1
):
    # Normalize the embeddings to prevent scale issues affecting the cosine distance
    prediction_norm = F.normalize(prediction, p=2, dim=1)
    target_norm = F.normalize(target, p=2, dim=1)
    non_targets_norm = F.normalize(non_targets, p=2, dim=-1)

    # Calculate the cosine similarity between predictions and targets
    cosine_similarity = (prediction_norm * target_norm).sum(dim=1)

    # Calculate the positive part of the loss (distance from the correct targets)
    positive_loss = 1 - cosine_similarity

    # Calculate the negative part of the loss (push away from incorrect targets by a margin)
    # Compute cosine similarities between each prediction and all its non-targets
    negative_cosine_sim = torch.bmm(non_targets_norm, prediction_norm.unsqueeze(-1)).squeeze(-1)

    # Apply the margin and ReLU to ensure only positive contributions to the loss
    negative_loss = F.relu(negative_cosine_sim + margin - cosine_similarity.unsqueeze(1)).mean(dim=1)

    # The final loss is a sum of the positive loss and the average negative loss
    total_loss = positive_loss.mean() + negative_loss.mean()
    return total_loss


def loss_func(prediction: torch.Tensor, target: torch.Tensor, mse: bool = False, cosine: bool = True) -> torch.Tensor:
    """Loss function for MuZero agent's value and reward."""
    # logging.debug("prediction: ", prediction)
    # logging.debug("target: ", target)
    assert prediction.shape == target.shape

    if not mse:
        assert len(prediction.shape) == 2

    if mse:
        # [B]
        # return 0.5 * torch.square((prediction - target))
        return F.mse_loss(prediction, target, reduction='none')

    if cosine:
        if prediction.isnan().any():
            logging.debug("prediction: ", prediction)
        if target.isnan().any():
            logging.debug("target: ", target)
        # logging.debug("target length: ", torch.sqrt(torch.sum(target ** 2, axis=-1)))
        # logging.debug("prediction length: ", torch.sqrt(torch.sum(prediction ** 2, axis=-1)))
        l = F.cosine_embedding_loss(prediction, target, torch.ones(target.shape[0], device=prediction.device))
        # logging.debug("cosine loss: ", l)
        return l

    # [B]
    # return torch.sum(-target * F.log_softmax(prediction, dim=1), dim=1)
    return F.cross_entropy(prediction, target, reduction='none')


def compute_n_step_target(
    rewards: List[float], root_values: List[float], td_steps: int, discount: float
) -> List[float]:
    """Compute n-step target for Atari and classic openAI Gym problems.

    zt = ut+1 + γut+2 + ... + γn−1ut+n + γnνt+n

    Args:
        rewards: a list of rewards received from the env, length T.
        root_values: a list of root node value from MCTS search, length T.
        td_steps: the number of steps into the future for n-step value.
        discount: discount for future reward.

    Returns:
        a list of n-step target value, length T.

    Raises:
        ValueError:
            lists `rewards` and `root_values` do not have equal length.
    """

    if len(rewards) != len(root_values):
        raise ValueError('Arguments `rewards` and `root_values` don have the same length.')

    T = len(rewards)

    # Make a shallow copy to avoid manipulate on the original data.
    _rewards = list(rewards)
    _root_values = list(root_values)

    # Padding zeros at the end of trajectory for easy computation.
    _rewards += [0] * td_steps
    _root_values += [0] * td_steps

    target_values = []
    for t in range(T):
        bootstrap_index = t + td_steps
        value = sum([discount**i * reward for i, reward in enumerate(_rewards[t:bootstrap_index])])

        # Add MCTS root node search value for bootstrap.
        value += discount**td_steps * _root_values[bootstrap_index]
        target_values.append(value)

    return target_values


def compute_mc_return_target(rewards: List[float], player_ids: List[float]) -> List[float]:
    """Compute the target value using Monte Carlo returns. This is for board game only.

    Args:
        rewards: a list of rewards received from the env, length T.
        player_ids: a list of player id for each of the transition, length T.

    Returns:
        a list of target value using MC return, length T.

    Raises:
        ValueError:
            lists `rewards` and `player_ids` do not have equal length.
    """
    if len(rewards) != len(player_ids):
        raise ValueError('Arguments `rewards` and `player_ids` don have the same length.')

    T = len(rewards)

    target_values = [0.0] * T

    final_reward = rewards[-1]
    final_player = player_ids[-1]

    if final_reward != 0.0:
        for t in range(T):
            if player_ids[t] == final_player:
                target_values[t] = final_reward
            else:
                target_values[t] = -final_reward

    return target_values


def make_unroll_sequence(
    observations: List[np.ndarray],
    actions: List[torch.Tensor],
    rewards: List[float],
    pi_probs: List[np.ndarray],
    values: List[float],
    priorities: List[float],
    unroll_steps: int,
) -> Iterable[tuple[Transition, float]]:
    """Turn a lists of episode history into a list of structured transition object,
    and stack unroll_steps for actions, rewards, values, MCTS policy.

    Args:
        observations: a list of history environment observations.
        actions: a list of history actual actions taken in the environment.
        rewards: a list of history reward received from the environment.
        pi_probs: a list of history policy probabilities from the MCTS search result.
        values: a list of n-step target value.
        priorities: a list of priorities for each transition.
        unroll_steps: number of unroll steps during traning.

    Returns:
        yeilds tuple of structured Transition object and the associated priority for the specific transition.

    """

    T = len(observations)

    # States past the end of games are treated as absorbing states.
    if len(actions) == T:
        actions += [torch.zeros_like(actions[0], dtype=actions[0].dtype)] * unroll_steps
    if len(rewards) == T:
        rewards += [0] * unroll_steps
    if len(values) == T:
        values += [0] * unroll_steps
    if len(pi_probs) == T:
        absorb_policy = np.ones_like(pi_probs[-1]) / len(pi_probs[-1])
        pi_probs += [absorb_policy] * unroll_steps

    assert len(actions) == len(rewards) == len(values) == len(pi_probs) == T + unroll_steps

    for t in range(T):
        end_index = t + unroll_steps
        action_sequence = torch.stack([action.cpu().float() for action in actions[t:end_index]])
        stacked_action = action_sequence.numpy()
        stacked_reward = np.array(rewards[t:end_index], dtype=np.float32)
        stacked_value = np.array(values[t:end_index], dtype=np.float32)
        stacked_pi_prob = np.array(pi_probs[t:end_index], dtype=np.float32)
        transition = Transition(
            state=observations[t],  # no stacking for observation, since it is only used to get initial hidden state.
            action=stacked_action,
            reward=stacked_reward,
            value=stacked_value,
            pi_prob=stacked_pi_prob,
        )
        logging.debug(f"transition: {transition}")
        yield (
            transition,
            priorities[t],
        )


def _make_unroll_sequence(
    observations: List[np.ndarray],
    actions: List[int],
    rewards: List[float],
    pi_probs: List[np.ndarray],
    values: List[float],
    priorities: List[float],
    unroll_steps: int,
) -> Iterable[Transition]:
    """Turn a lists of episode history into a list of structured transition object,
    and stack unroll_steps for actions, rewards, values, MCTS policy.

    Args:
        observations: a list of history environment observations.
        actions: a list of history actual actions taken in the environment.
        rewards: a list of history reward received from the environment.
        pi_probs: a list of history policy probabilities from the MCTS search result.
        values: a list of n-step target value.
        priorities: a list of priorities for each transition.
        unroll_steps: number of unroll steps during traning.

    Returns:
        yeilds tuple of structured Transition object and the associated priority for the specific transition.

    """

    T = len(observations)

    # States past the end of games are treated as absorbing states.
    if len(actions) == T:
        actions += [0] * unroll_steps
    if len(rewards) == T:
        rewards += [0] * unroll_steps
    if len(values) == T:
        values += [0] * unroll_steps
    if len(pi_probs) == T:
        absorb_policy = np.ones_like(pi_probs[-1]) / len(pi_probs[-1])
        pi_probs += [absorb_policy] * unroll_steps

    assert len(actions) == len(rewards) == len(values) == len(pi_probs) == T + unroll_steps

    for t in range(T):
        end_index = t + unroll_steps
        stacked_action = np.array(actions[t:end_index], dtype=np.float32)
        stacked_reward = np.array(rewards[t:end_index], dtype=np.float32)
        stacked_value = np.array(values[t:end_index], dtype=np.float32)
        stacked_pi_prob = np.array(pi_probs[t:end_index], dtype=np.float32)

        yield (
            Transition(
                state=observations[
                    t
                ],  # no stacking for observation, since it is only used to get initial hidden state.
                action=stacked_action,
                reward=stacked_reward,
                value=stacked_value,
                pi_prob=stacked_pi_prob,
            ),
            priorities[t],
        )


def handle_exit_signal():
    """Listen to exit signal like ctrl-c or kill from os and try to exit the process forcefully."""

    def shutdown(signal_code, frame):
        del frame
        logging.info(
            f'Received signal {signal_code}: terminating process...',
        )
        sys.exit(128 + signal_code)

    # Listen to signals to exit process.
    signal.signal(signal.SIGABRT, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    # signal.signal(signal.SIGTERM, shutdown)


def get_time_stamp(as_file_name: bool = False) -> str:
    t = time.localtime()
    if as_file_name:
        timestamp = time.strftime('%Y%m%d_%H%M%S', t)
    else:
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', t)
    return timestamp


def create_checkpoint(state_to_save: Mapping[Text, Any], ckpt_file: str) -> None:
    torch.save(state_to_save, ckpt_file)


def load_checkpoint(ckpt_file: str, device: torch.device) -> Mapping[Text, Any]:
    return torch.load(ckpt_file, map_location=torch.device(device))


def save_to_file(obj: Any, file_name: str) -> None:
    """Save object to file."""
    pickle.dump(obj, open(file_name, 'wb'))


def load_from_file(file_name: str) -> Any:
    """Load object from file."""
    return pickle.load(open(file_name, 'rb'))


def disable_auto_grad(network: torch.nn.Module) -> None:
    for p in network.parameters():
        p.requires_grad = False
