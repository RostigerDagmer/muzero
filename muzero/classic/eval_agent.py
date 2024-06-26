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
from absl import app
from absl import flags
from absl import logging
import os
import torch

from muzero.network import MuZeroMLPNet
from muzero.gym_env import create_classic_environment, record_video_env
from muzero.pipeline import load_checkpoint
from muzero.mcts import uct_search
from muzero.config import make_classic_config

FLAGS = flags.FLAGS
flags.DEFINE_string('environment_name', 'CartPole-v1', "Classic problem like 'CartPole-v1', 'LunarLander-v2'")
flags.DEFINE_integer('stack_history', 4, 'Stack previous states.')

flags.DEFINE_integer('seed', 5, 'Seed the runtime.')

flags.DEFINE_string(
    'load_checkpoint_file',
    # 'saved_checkpoints/CartPole-v1_train_steps_44800',
    'checkpoints/CartPole-v1_train_steps_100000_final',
    # 'saved_checkpoints/LunarLander-v2_train_steps_58400',
    'Load the checkpoint from file.',
)
flags.DEFINE_string('record_video_dir', 'recordings', 'Record play video.')


def main(argv):
    """Evaluates MuZero agent on classic control problem."""
    del argv

    runtime_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    eval_env = create_classic_environment(FLAGS.environment_name, FLAGS.seed, FLAGS.stack_history)
    input_shape = eval_env.observation_space.shape
    num_actions = eval_env.action_space.n

    config = make_classic_config()

    network = MuZeroMLPNet(
        input_shape, num_actions, config.num_planes, config.value_support_size, config.reward_support_size, config.hidden_dim
    )

    # Load states from checkpoint to resume training.
    if FLAGS.load_checkpoint_file is not None and os.path.isfile(FLAGS.load_checkpoint_file):
        loaded_state = load_checkpoint(FLAGS.load_checkpoint_file, 'cpu')
        network.load_state_dict(loaded_state['network'])
        logging.info(f'Loaded state from checkpoint {FLAGS.load_checkpoint_file}')

    network.eval().to(runtime_device)

    if FLAGS.record_video_dir is not None and FLAGS.record_video_dir != '':
        eval_env = record_video_env(eval_env, FLAGS.record_video_dir)

    steps = 0
    returns = 0.0

    obs = eval_env.reset()
    while True:
        action, *_ = uct_search(
            state=obs,
            network=network,
            device=runtime_device,
            config=config,
            temperature=0.0,
            actions_mask=eval_env.actions_mask,
            current_player=eval_env.current_player,
            opponent_player=eval_env.opponent_player,
            deterministic=True,
        )

        obs, reward, done, _ = eval_env.step(action)
        steps += 1
        returns += reward

        if done:
            break

    eval_env.close()
    logging.info(f'Episode returns: {returns}, steps: {steps}')


if __name__ == '__main__':
    app.run(main)
