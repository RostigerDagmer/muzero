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
from absl import app
from absl import flags
import multiprocessing
import threading
import gym
import numpy as np
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

FLAGS = flags.FLAGS
flags.DEFINE_string('environment_name', 'CartPole-v1', "Classic problem like 'CartPole-v1', 'LunarLander-v2'")
flags.DEFINE_integer('stack_history', 4, 'Stack last N states and actions.')
flags.DEFINE_integer('num_actors', 6, 'Number of self-play actor processes.')
flags.DEFINE_integer('num_training_steps', 100000, 'Number of traning steps.')
flags.DEFINE_integer('batch_size', 128, 'Batch size for traning.')
flags.DEFINE_integer('replay_capacity', 50000, 'Maximum replay size.')
flags.DEFINE_integer('min_replay_size', 5000, 'Minimum replay size before start to do traning.')
flags.DEFINE_float(
    'priority_exponent', 1.0, 'Priotiry exponent used in prioritized replay, 0 means using uniform random replay.'
)
flags.DEFINE_float('importance_sampling_exponent', 1.0, 'Importance sampling exponent value.')

flags.DEFINE_integer('seed', 1, 'Seed the runtime.')

flags.DEFINE_bool('use_tensorboard', True, 'Monitor performance with Tensorboard, default on.')
flags.DEFINE_bool('clip_grad', False, 'Clip gradient, default off.')
flags.DEFINE_string('checkpoint_dir', 'checkpoints', 'Path for save checkpoint files.')
flags.DEFINE_integer(
    'samples_save_frequency',
    -1,
    'The frequency (measured in number added in replay) to save self-play samples in replay, default -1 do not save.',
)
flags.DEFINE_string(
    'load_checkpoint_file',
    'checkpoints/CartPole-v1_train_steps_1000',
    'Load the checkpoint from file.',
)
flags.DEFINE_string('samples_save_dir', 'samples', 'Path for save self-play samples in replay to file.')
flags.DEFINE_string('tag', '', 'Add tag to Tensorboard log file.')



class ActionEncoderWith():
    def __init__(self, action_embeddings: torch.Tensor):
        self.action_embeddings = action_embeddings

    def __call__(self, actions: int):
        return self.action_embeddings[actions]

    def get_action_embeddings(self):
        return self.action_embeddings
    
def use_distance(action_index: float, distance: float) -> torch.Tensor:
    # distance can be in range [-1, 1]
    # action index is {0, 1}
    # we need to map (0, 0) to (0)
    # (1, 0) to (1)
    # (0, -1) -> (1)
    # (1, -1) -> (0)

    centered = action_index * 2 - 1
    mapped = centered * distance
    scaled = (mapped + 1) / 2

    logging.debug("mapped action: ", action_index)
    logging.debug("with distance: ", distance)
    logging.debug("to scalar: ", scaled)

    return torch.Tensor([scaled])

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
    eval_env, actions = create_classic_environment(
        FLAGS.environment_name, FLAGS.seed + 2, FLAGS.stack_history, output_actions=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/pythia-70m-deduped",
        revision="step3000",
        device_map="cpu",
    )
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenized_actions = tokenizer(actions, return_tensors="pt", padding=True, truncation=True)
    # input_shape = self_play_envs[0].observation_space.shape
    # num_actions = self_play_envs[0].action_space.n
    input_ids = tokenized_actions['input_ids'].to('cpu')
    attention_mask = tokenized_actions['attention_mask'].to('cpu')
    action_encoder = ContinousActionEncoder(device='cpu')
    action_embeddings = action_encoder(input_ids, attention_mask)

    print("action shapes: ", action_embeddings.shape)

    action_decoder = ContinousActionDecoder(action_embeddings.clone())

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
        action_embeddings.shape[-1],
        VitConfig(),
        512,
        config.num_planes,
        config.value_support_size,
        config.reward_support_size,
        FLAGS.stack_history,
        8,  # attention heads
        ContinousEncoderHead.pythia,
    )

    optimizer = None  # torch.optim.Adam(network.parameters(), lr=config.lr_init, weight_decay=config.weight_decay)
    lr_scheduler = None  # MultiStepLR(optimizer, milestones=config.lr_milestones, gamma=config.lr_decay_rate)

    action_encoder = ContinousActionEncoder(device='cpu')
    action_decoder = ContinousActionDecoder(action_embeddings.clone())

    actor_network = ContinousMuzeroNet(
        action_encoder,
        action_decoder,
        action_embeddings.shape[-1],
        VitConfig(),
        512,
        config.num_planes,
        config.value_support_size,
        config.reward_support_size,
        FLAGS.stack_history,
        8,  # attention heads
        ContinousEncoderHead.pythia,
    )

    actor_network.share_memory()

    action_encoder = ContinousActionEncoder(device='cpu')
    action_decoder = ContinousActionDecoder(action_embeddings.clone())

    new_ckpt_network = ContinousMuzeroNet(
        action_encoder,
        action_decoder,
        action_embeddings.shape[-1],
        VitConfig(),
        512,
        config.num_planes,
        config.value_support_size,
        config.reward_support_size,
        FLAGS.stack_history,
        8,  # attention heads
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

    action_encoder = ContinousActionEncoder(device='cpu')
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
            1,
            False,
            use_distance,
            action_decoder,
            action_encoder,
        ),
    )
    evaluator.start()

    action_encoder = ActionEncoderWith(action_embeddings.clone())

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
                use_distance,
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
