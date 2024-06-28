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
"""Runs MuZero self-play training pipeline on Atari game."""
from typing import Optional
from absl import app
from absl import flags
from absl import logging
import os
import multiprocessing
import threading
import numpy as np
import torch

from muzero.network import MuZeroAtariNet
from muzero.continous.io import ContinousActionDecoder, ContinousActionEncoder, VitConfig
from muzero.continous.net import ContinousMuzeroNet
from muzero.replay import PrioritizedReplay
from muzero.config import make_atari_config
from muzero.gym_env import create_atari_environment
from muzero.pipeline import run_self_play, run_training, run_data_collector, run_evaluator, load_checkpoint, load_from_file
from pydantic import BaseModel

class EnvironmentConfig(BaseModel):
    name: str = 'Breakout'
    height: int = 224
    width: int = 224
    frame_skip: int = 4
    frame_stack: int = 16
    max_episode_steps: int = 108000
    gray_scale: bool = False
    clip_reward: bool = True
    done_on_life_loss: bool = True
    num_actors: int = 4

class TrainingConfig(BaseModel):
    num_training_steps: int = int(10e6)
    batch_size: int = 128
    replay_capacity: int = int(1e6)
    min_replay_size: int = 1000
    priority_exponent: float = 1.0
    importance_sampling_exponent: float = 1.0

class RuntimeConfig(BaseModel):
    seed: int = 1
    use_tensorboard: bool = True
    clip_grad: bool = True

class CheckpointConfig(BaseModel):
    checkpoint_dir: str = 'checkpoints'
    load_checkpoint_file: Optional[str] = None

class SamplesConfig(BaseModel):
    save_frequency: int = -1
    save_dir: str = 'samples'
    load_samples_file: Optional[str] = ''
    tag: Optional[str] = ''


class Config(BaseModel):
    environment: EnvironmentConfig = EnvironmentConfig()
    training: TrainingConfig = TrainingConfig()
    runtime: RuntimeConfig = RuntimeConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    samples: SamplesConfig = SamplesConfig()


class ActionEncoderWith():
    def __init__(self, action_embeddings):
        self.action_embeddings = action_embeddings

    def __call__(self, actions):
        return self.action_embeddings[actions]

    def get_action_embeddings(self):
        return self.action_embeddings

def main(argv):
    """Trains MuZero agent on Atari games."""
    del argv
    config = Config()
    # runtime_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    runtime_device = torch.device('mps')
    random_state = np.random.RandomState(config.runtime.seed)  # pylint: disable=no-member
    torch.manual_seed(config.runtime.seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def environment_builder():
        return create_atari_environment(
            env_name=config.environment.name,
            screen_height=config.environment.height,
            screen_width=config.environment.width,
            frame_skip=config.environment.frame_skip,
            frame_stack=config.environment.frame_stack,
            max_episode_steps=config.environment.max_episode_steps,
            seed=random_state.randint(1, 2**31),
            noop_max=30,
            terminal_on_life_loss=False,
            clip_reward=False,
            output_actions=True,
            resize_and_gray=False
        )

    self_play_envs, self_play_actions = zip(*[environment_builder() for _ in range(config.environment.num_actors)])

    eval_env, eval_actions = environment_builder()

    input_shape = self_play_envs[0].observation_space.shape
    num_actions = self_play_envs[0].action_space.n

    tag = self_play_envs[0].spec.id
    if config.samples.tag is not None and config.samples.tag != '':
        tag = f'{tag}_{config.samples.tag}'

    atari_config = make_atari_config(
        num_training_steps=config.training.num_training_steps,
        batch_size=config.training.batch_size,
        min_replay_size=config.training.min_replay_size,
        use_tensorboard=config.runtime.use_tensorboard,
        clip_grad=config.runtime.clip_grad,
    )
    
    formatted_actions = [f"action: {action}" for action in eval_actions]
    print(f"formatted actions: {formatted_actions}")
    
    # huggingface tokenizer
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # tokenized_actions = tokenizer(formatted_actions, padding=True, return_tensors="pt")
    tokenized_actions = tokenizer(formatted_actions)

    # print(f"tokenized actions: {tokenized_actions}")
    action_encoder = ContinousActionEncoder()
    action_embeddings = action_encoder(tokenized_actions, None)

    print("action embeddings shape: ", action_embeddings.shape)
    
    action_decoder = ContinousActionDecoder(action_embeddings.clone())

    network = ContinousMuzeroNet(
        action_encoder,
        action_decoder,
        action_embeddings.shape[-1],
        VitConfig(),
        512,
        atari_config.num_planes,
        atari_config.value_support_size,
        atari_config.reward_support_size,
        config.environment.frame_stack,
        8, # attention heads
    )

    action_decoder = ContinousActionDecoder(action_embeddings.clone())
    actor_network = ContinousMuzeroNet(
        action_encoder,
        action_decoder,
        action_embeddings.shape[-1],
        VitConfig(),
        512,
        atari_config.num_planes,
        atari_config.value_support_size,
        atari_config.reward_support_size,
        config.environment.frame_stack,
        8, # attention heads
    )
    
    action_decoder = ContinousActionDecoder(action_embeddings.clone())
    actor_network.share_memory()

    new_ckpt_network = ContinousMuzeroNet(
        action_encoder,
        action_decoder,
        action_embeddings.shape[-1],
        VitConfig(),
        512,
        atari_config.num_planes,
        atari_config.value_support_size,
        atari_config.reward_support_size,
        config.environment.frame_stack,
        8, # attention heads
    )

    # list devices that the networks are on
    # print(f"network device: {[p.device for p in network.parameters()]}")
    # print(f"actor network device: {[p.device for p in actor_network.parameters()]}")
    # print(f"new ckpt network device: {[p.device for p in new_ckpt_network.parameters()]}")

    # print(f"encoder device net: {network.action_encoder.action_encoder.device}")
    # print(f"encoder device action: {actor_network.action_encoder.action_encoder.device}")
    # print(f"encoder device new_ckpt: {new_ckpt_network.action_encoder.action_encoder.device}")
    # print(f"decoder device net: {[t.device for t in network.action_decoder.action_set]}")
    # print(f"decoder device action: {[t.device for t in actor_network.action_decoder.action_set]}")
    # print(f"decoder device new_ckpt: {[t.device for t in new_ckpt_network.action_decoder.action_set]}")

    replay = PrioritizedReplay(
        config.training.replay_capacity,
        config.training.priority_exponent,
        config.training.importance_sampling_exponent,
        random_state,
    )

    # Use the stop_event to signaling actors to stop running.
    stop_event = multiprocessing.Event()
    # Transfer samples from self-play process to training process.
    data_queue = multiprocessing.SimpleQueue()
    # A shared list to store most recent new checkpoint files.
    manager = multiprocessing.Manager()
    checkpoint_files = manager.list()

    # Shared training steps counter, so actors can adjust temperature used in MCTS.
    train_steps_counter = multiprocessing.Value('i', 0)

    # Load states from checkpoint to resume training.
    if config.checkpoint.load_checkpoint_file is not None and os.path.isfile(config.checkpoint.load_checkpoint_file):
        loaded_state = load_checkpoint(config.checkpoint.load_checkpoint_file, 'cpu')
        network.load_state_dict(loaded_state['network'])
        optimizer_state = loaded_state['optimizer']
        scheduler_state = loaded_state['lr_scheduler']
        train_steps_counter.value = loaded_state['train_steps']

        actor_network.load_state_dict(loaded_state['network'])

        logging.info(f'Loaded state from checkpoint {config.checkpoint.load_checkpoint_file}')
        logging.info(f'Current state: train steps {train_steps_counter.value}, learing rate {scheduler_state.get("last_lr")}')
    else:
        logging.warn(f"could not load checkpoint file {config.checkpoint.load_checkpoint_file}")
        optimizer_state = None
        scheduler_state = None

    # Load replay samples
    if config.samples.load_samples_file is not None and os.path.isfile(config.samples.load_samples_file):
        try:
            replay.reset()
            replay_state = load_from_file(config.samples.load_samples_file)
            replay.set_state(replay_state)
            logging.info(f"Loaded replay samples from file '{config.samples.load_samples_file}'")
        except Exception:
            pass

    # Start to collect samples from self-play on a new thread.
    data_collector = threading.Thread(
        target=run_data_collector,
        args=(data_queue, replay, config.samples.save_frequency, config.samples.save_dir, tag),
    )
    data_collector.start()

    # Start the main training loop on a new thread.
    learner = threading.Thread(
        target=run_training,
        args=(
            config,
            network,
            optimizer_state,
            scheduler_state,
            runtime_device,
            actor_network,
            replay,
            data_queue,
            train_steps_counter,
            config.checkpoint.checkpoint_dir,
            checkpoint_files,
            stop_event,
            tag,
        ),
    )
    learner.start()
    
    action_encoder = ActionEncoderWith(action_embeddings.clone())

    # Start evaluation loop on a seperate process.
    evaluator = multiprocessing.Process(
        target=run_evaluator,
        args=(
            config,
            new_ckpt_network,
            runtime_device,
            eval_env,
            0.0,
            checkpoint_files,
            stop_event,
            tag,
            True, # override mask
            action_decoder,
            action_encoder,
        ),
    )
    evaluator.start()

    # # Start self-play processes.
    actors = []
    for i in range(config.environment.num_actors):
        action_decoder = ContinousActionDecoder(action_embeddings.clone())
        net_config = {
            "action_encoder": ContinousActionEncoder(),
            "action_decoder": action_decoder,
            "action_space_dim": action_embeddings.shape[-1],
            "vit_config": VitConfig(),
            "num_planes": atari_config.num_planes,
            "value_support_size": atari_config.value_support_size,
            "reward_support_size": atari_config.reward_support_size,
            "sequence_length": config.environment.frame_stack,
            "attention_heads": 8, # attention heads
        }
        actor = multiprocessing.Process(
            target=run_self_play,
            args=(
                config,
                i,
                net_config,
                runtime_device,
                self_play_envs[i],
                data_queue,
                train_steps_counter,
                stop_event,
                tag,
                True,
                action_decoder,
                action_encoder,
            ),
        )
        actor.start()
        actors.append(actor)

    for actor in actors:
        actor.join()
        actor.close()

    learner.join()
    data_collector.join()
    evaluator.join()


if __name__ == '__main__':
    # Set multiprocessing start mode
    multiprocessing.set_start_method('spawn')
    app.run(main)
