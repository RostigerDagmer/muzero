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
"""Runs MuZero self-play training pipeline on classic control problem like CartPole and LunarLander.
"""
from typing import Optional
from absl import app
from absl import flags
import multiprocessing
import threading
import gym
import numpy as np
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer
import logging

from muzero.continous.debug import NoMatplotFilter
from muzero.continous.net import (
    ContinousMuzeroNet,
    ContinousEncoderHead,
    VitConfig,
)

from muzero.continous.io import ContinousActionDecoder, ContinousActionEncoder
from muzero.replay import PrioritizedReplay, Transition
from muzero.config import make_classic_config
from muzero.gym_env import create_classic_environment
from muzero.pipeline import run_self_play, run_training, run_data_collector, run_evaluator

logging.basicConfig(level=logging.INFO)
for handler in logging.root.handlers:
    handler.addFilter(NoMatplotFilter())

class Flags(BaseModel):
    environment_name: str = 'CartPole-v1'
    stack_history: int = 4
    num_actors: int = 6
    num_training_steps: int = 100000
    batch_size: int = 128
    replay_capacity: int = 50000
    min_replay_size: int = 1000
    priority_exponent: float = 0.5
    importance_sampling_exponent: float = 0.4
    seed: int = 1
    use_tensorboard: bool = True
    clip_grad: bool = False
    checkpoint_dir: str = 'checkpoints'
    checkpoint_file: Optional[str] = None
    samples_save_frequency: int = -1
    samples_save_dir: str = 'samples'
    tag: str = ''

FLAGS = Flags()

class ActionEncoderWith():
    def __init__(self, action_embeddings: torch.Tensor):
        self.action_embeddings = action_embeddings

    def __call__(self, actions: int):
        return self.action_embeddings[actions]

    def get_action_embeddings(self):
        return self.action_embeddings

def main(argv):
    """Trains MuZero agent on classic control problems."""
    del argv

    runtime_device = 'mps' # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random_state = np.random.RandomState(FLAGS.seed)  # pylint: disable=no-member
    torch.manual_seed(FLAGS.seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    self_play_envs: list[gym.Env] = [
        create_classic_environment(FLAGS.environment_name, FLAGS.seed + i**2, FLAGS.stack_history)
        for i in range(FLAGS.num_actors)
    ]
    eval_env = create_classic_environment(
        FLAGS.environment_name, FLAGS.seed + 2, FLAGS.stack_history
    )

    num_actions = self_play_envs[0].action_space.n

    action_decoder = None
    action_encoder = None

    tag = self_play_envs[0].spec.id
    if FLAGS.tag is not None and FLAGS.tag != '':
        tag = f'{tag}_{FLAGS.tag}'

    config = make_classic_config(
        num_training_steps=FLAGS.num_training_steps,
        batch_size=FLAGS.batch_size,
        min_replay_size=FLAGS.min_replay_size,
        use_tensorboard=FLAGS.use_tensorboard,
        clip_grad=FLAGS.clip_grad,
    )

    network = ContinousMuzeroNet(
        action_encoder,
        action_decoder,
        num_actions,
        VitConfig(),
        512,
        config.num_planes,
        config.value_support_size,
        config.reward_support_size,
        FLAGS.stack_history,
        4,  # attention heads
        ContinousEncoderHead.pythia,
    )

    optimizer = None  # torch.optim.Adam(network.parameters(), lr=config.lr_init, weight_decay=config.weight_decay)
    lr_scheduler = None  # MultiStepLR(optimizer, milestones=config.lr_milestones, gamma=config.lr_decay_rate)

    actor_network = ContinousMuzeroNet(
        action_encoder,
        action_decoder,
        num_actions,
        VitConfig(),
        512,
        config.num_planes,
        config.value_support_size,
        config.reward_support_size,
        FLAGS.stack_history,
        4,  # attention heads
        ContinousEncoderHead.pythia,
    )

    actor_network.share_memory()

    new_ckpt_network = ContinousMuzeroNet(
        action_encoder,
        action_decoder,
        num_actions,
        VitConfig(),
        512,
        config.num_planes,
        config.value_support_size,
        config.reward_support_size,
        FLAGS.stack_history,
        4,  # attention heads
        ContinousEncoderHead.pythia,
    )

    replay: PrioritizedReplay[Transition] = PrioritizedReplay(
        FLAGS.replay_capacity,
        FLAGS.priority_exponent,
        FLAGS.importance_sampling_exponent,
        random_state,
    )

    # Use the stop_event to signaling actors to stop running.
    stop_event = multiprocessing.Event()
    # Transfer samples from self-play process to training process.
    data_queue: multiprocessing.SimpleQueue[tuple[Transition, float]] = multiprocessing.SimpleQueue()
    # A shared list to store most recent new checkpoint files.
    manager = multiprocessing.Manager()
    checkpoint_files = manager.list()

    # Shared training steps counter, so actors can adjust temperature used in MCTS.
    train_steps_counter = multiprocessing.Value('i', 0)

    # Start to collect samples from self-play on a new thread.
    data_collector = threading.Thread(
        target=run_data_collector,
        args=(data_queue, replay, FLAGS.samples_save_frequency, FLAGS.samples_save_dir, tag),
    )
    data_collector.start()

    # Start the main training loop on a new thread.
    learner = threading.Thread(
        target=run_training,
        args=(
            config,
            network,
            optimizer,
            lr_scheduler,
            runtime_device,
            actor_network,
            replay,
            data_queue,
            train_steps_counter,
            FLAGS.checkpoint_dir,
            checkpoint_files,
            stop_event,
            tag,
        ),
    )
    learner.start()
    
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
            1,
            False,
            None,
            action_decoder,
            action_encoder,
        ),
    )
    evaluator.start()

    # # Start self-play processes.
    actors = []
    for i in range(FLAGS.num_actors):
        # net_config = {
        #     "action_encoder": ContinousActionEncoder(),
        #     "action_decoder": action_decoder,
        #     "action_space_dim": action_embeddings.shape[-1],
        #     "vit_config": VitConfig(),
        #     "num_planes": config.num_planes,
        #     "value_support_size": config.value_support_size,
        #     "reward_support_size": config.reward_support_size,
        #     "sequence_length": FLAGS.stack_history,
        #     "attention_heads": 8, # attention heads
        # }
        actor = multiprocessing.Process(
            target=run_self_play,
            args=(
                config,
                i,
                actor_network,
                runtime_device,
                self_play_envs[i],
                data_queue,
                train_steps_counter,
                stop_event,
                tag,
                False,
                None,
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
