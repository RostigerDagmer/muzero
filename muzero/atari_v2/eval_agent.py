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
from pathlib import Path
import gym
import torch

from muzero.network import MuZeroAtariNet
from muzero.pipeline import load_checkpoint
from muzero.mcts import uct_search
from muzero.config import make_atari_config
from muzero.gym_env import create_atari_environment, record_video_env
from muzero.continous import ContinousActionDecoder, ContinousActionEncoder, ContinousMuzeroNet, VitConfig, tokenizer


class ActionEncoderWith:
    def __init__(self, action_embeddings):
        self.action_embeddings = action_embeddings

    def __call__(self, actions):
        return self.action_embeddings[actions]

    def get_action_embeddings(self):
        return self.action_embeddings


FLAGS = flags.FLAGS
flags.DEFINE_string('environment_name', 'Breakout', 'Classic problem like Breakout, Pong')
flags.DEFINE_integer('screen_width', 224, 'Environment frame screen height.')
flags.DEFINE_integer('screen_height', 224, 'Environment frame screen height.')
flags.DEFINE_integer('stack_history', 16, 'Stack previous states.')
flags.DEFINE_integer('frame_skip', 4, 'Skip n frames.')
flags.DEFINE_bool('gray_scale', True, 'Gray scale observation image.')
flags.DEFINE_bool('clip_reward', True, 'Clip reward in the range [-1, 1], default on.')
flags.DEFINE_bool('done_on_life_loss', True, 'End of game if loss a life, default on.')

flags.DEFINE_integer('seed', 5, 'Seed the runtime.')

flags.DEFINE_string(
    'load_checkpoint_file',
    'checkpoints/BreakoutNoFrameskip-v4_train_steps_4000',
    'Load the checkpoint from file.',
)
flags.DEFINE_string('record_video_dir', 'recordings/classic', 'Record play video.')


def main(argv):
    """Evaluates MuZero agent on Atari games."""
    del argv

    runtime_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    eval_env, eval_actions = create_atari_environment(
        env_name=FLAGS.environment_name,
        screen_height=FLAGS.screen_height,
        screen_width=FLAGS.screen_width,
        frame_skip=FLAGS.frame_skip,
        frame_stack=FLAGS.stack_history,
        seed=1247097,
        noop_max=30,
        terminal_on_life_loss=True,
        clip_reward=False,
        output_actions=True,
        resize_and_gray=False,
    )
    input_shape = eval_env.observation_space.shape
    num_actions = eval_env.action_space.n

    config = make_atari_config()

    formatted_actions = [f"action: {action}" for action in eval_actions]
    print(f"formatted actions: {formatted_actions}")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenized_actions = tokenizer(formatted_actions, padding=True, return_tensors="pt")
    # print(f"tokenized actions: {tokenized_actions}")
    action_encoder = ContinousActionEncoder()
    action_embeddings = action_encoder(tokenized_actions.input_ids, tokenized_actions.attention_mask)

    print("action embeddings shape: ", action_embeddings.shape)

    action_decoder = ContinousActionDecoder(action_embeddings.clone())

    network = ContinousMuzeroNet(
        action_encoder,
        action_decoder,
        action_embeddings.shape[-1],
        VitConfig(),
        config.num_planes,
        config.value_support_size,
        config.reward_support_size,
        FLAGS.stack_history,
        8,  # attention heads
    )

    action_encoder = ActionEncoderWith(action_embeddings.clone())

    # Load states from checkpoint to resume training.
    if FLAGS.load_checkpoint_file is not None and os.path.isfile(FLAGS.load_checkpoint_file):
        loaded_state = load_checkpoint(FLAGS.load_checkpoint_file, 'cpu')
        network.load_state_dict(loaded_state['network'])
        logging.info(f'Loaded state from checkpoint {FLAGS.load_checkpoint_file}')

    network.eval()

    if FLAGS.record_video_dir is not None and FLAGS.record_video_dir != '':
        eval_env = record_video_env(eval_env, FLAGS.record_video_dir)

    steps = 0
    returns = 0.0

    obs = eval_env.reset()
    MAX_STEPS = 1000
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
            action_encoder=action_encoder,
            action_decoder=action_decoder,
        )
        print("env step: ", steps)
        obs, reward, done, _ = eval_env.step(action)
        steps += 1
        returns += reward

        if done or steps >= MAX_STEPS:
            break

    eval_env.close()
    logging.info(f'Episode returns: {returns}, steps: {steps}')


if __name__ == '__main__':
    app.run(main)
