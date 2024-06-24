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
from absl import app
from absl import flags
from absl import logging
import os
import multiprocessing
import threading
import numpy as np
import torch

from muzero.network import MuZeroAtariNet
from muzero.continous import ContinousActionDecoder, ContinousActionEncoder, ContinousMuzeroNet, VitConfig, tokenizer
from muzero.replay import PrioritizedReplay
from muzero.config import make_atari_config
from muzero.gym_env import create_atari_environment
from muzero.pipeline import run_self_play, run_training, run_data_collector, run_evaluator, load_checkpoint, load_from_file

FLAGS = flags.FLAGS
flags.DEFINE_string('environment_name', 'Breakout', 'Classic problem like Breakout, Pong etc.')
flags.DEFINE_integer('environment_height', 224, 'Environment frame screen height.')
flags.DEFINE_integer('environment_width', 224, 'Environment frame screen width.')
flags.DEFINE_integer('environment_frame_skip', 4, 'Number of frames to skip.')
flags.DEFINE_integer('environment_frame_stack', 16, 'Number of frames to stack.')
flags.DEFINE_integer('max_episode_steps', 108000, 'Maximum steps (before frame skip) per episode.')
flags.DEFINE_bool('gray_scale', False, 'Gray scale observation image.')
flags.DEFINE_bool('clip_reward', True, 'Clip reward in the range [-1, 1], default on.')
flags.DEFINE_bool('done_on_life_loss', True, 'End of game if loss a life, default on.')
flags.DEFINE_integer('num_actors', 4, 'Number of self-play actor processes.')

flags.DEFINE_integer('num_training_steps', int(10e6), 'Number of traning steps.')
flags.DEFINE_integer('batch_size', 128, 'Batch size for training.')
flags.DEFINE_integer('replay_capacity', int(1e6), 'Maximum replay size.')
flags.DEFINE_integer('min_replay_size', 1000, 'Minimum replay size before start to do traning.')
flags.DEFINE_float(
    'priority_exponent', 1.0, 'Priotiry exponent used in prioritized replay, 0 means using uniform random replay.'
)
flags.DEFINE_float('importance_sampling_exponent', 1.0, 'Importance sampling exponent value.')

flags.DEFINE_integer('seed', 1, 'Seed the runtime.')
flags.DEFINE_bool('use_tensorboard', True, 'Monitor performance with Tensorboard, default on.')
flags.DEFINE_bool('clip_grad', True, 'Clip gradient, default off.')

flags.DEFINE_string('checkpoint_dir', 'checkpoints', 'Path for save checkpoint files.')
flags.DEFINE_string(
    'load_checkpoint_file',
    'checkpoints/BreakoutNoFrameskip-v4_train_steps_1000',
    # '',
    'Load the checkpoint from file.',
)

flags.DEFINE_integer(
    'samples_save_frequency',
    -1,
    'The frequency (measured in number added in replay) to save self-play samples in replay, default -1 do not save.',
)
flags.DEFINE_string('samples_save_dir', 'samples', 'Path for save self-play samples in replay to file.')
flags.DEFINE_string('load_samples_file', '', 'Load the replay samples from file.')
flags.DEFINE_string('tag', '', 'Add tag to Tensorboard log file.')


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

    # runtime_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    runtime_device = torch.device('mps')
    random_state = np.random.RandomState(FLAGS.seed)  # pylint: disable=no-member
    torch.manual_seed(FLAGS.seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def environment_builder():
        return create_atari_environment(
            env_name=FLAGS.environment_name,
            screen_height=FLAGS.environment_height,
            screen_width=FLAGS.environment_width,
            frame_skip=FLAGS.environment_frame_skip,
            frame_stack=FLAGS.environment_frame_stack,
            max_episode_steps=FLAGS.max_episode_steps,
            seed=random_state.randint(1, 2**31),
            noop_max=30,
            terminal_on_life_loss=False,
            clip_reward=False,
            output_actions=True,
            resize_and_gray=False
        )

    self_play_envs, self_play_actions = zip(*[environment_builder() for _ in range(FLAGS.num_actors)])

    eval_env, eval_actions = environment_builder()

    input_shape = self_play_envs[0].observation_space.shape
    num_actions = self_play_envs[0].action_space.n

    tag = self_play_envs[0].spec.id
    if FLAGS.tag is not None and FLAGS.tag != '':
        tag = f'{tag}_{FLAGS.tag}'

    config = make_atari_config(
        num_training_steps=FLAGS.num_training_steps,
        batch_size=FLAGS.batch_size,
        min_replay_size=FLAGS.min_replay_size,
        use_tensorboard=FLAGS.use_tensorboard,
        clip_grad=FLAGS.clip_grad,
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
        config.num_planes,
        config.value_support_size,
        config.reward_support_size,
        FLAGS.environment_frame_stack,
        8, # attention heads
    )

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
        FLAGS.environment_frame_stack,
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
        config.num_planes,
        config.value_support_size,
        config.reward_support_size,
        FLAGS.environment_frame_stack,
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
        FLAGS.replay_capacity,
        FLAGS.priority_exponent,
        FLAGS.importance_sampling_exponent,
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
    if FLAGS.load_checkpoint_file is not None and os.path.isfile(FLAGS.load_checkpoint_file):
        loaded_state = load_checkpoint(FLAGS.load_checkpoint_file, 'cpu')
        network.load_state_dict(loaded_state['network'])
        optimizer_state = loaded_state['optimizer']
        scheduler_state = loaded_state['lr_scheduler']
        train_steps_counter.value = loaded_state['train_steps']

        actor_network.load_state_dict(loaded_state['network'])

        logging.info(f'Loaded state from checkpoint {FLAGS.load_checkpoint_file}')
        logging.info(f'Current state: train steps {train_steps_counter.value}, learing rate {scheduler_state.get("last_lr")}')
    else:
        logging.warn(f"could not load checkpoint file {FLAGS.load_checkpoint_file}")
        optimizer_state = None
        scheduler_state = None

    # Load replay samples
    if FLAGS.load_samples_file is not None and os.path.isfile(FLAGS.load_samples_file):
        try:
            replay.reset()
            replay_state = load_from_file(FLAGS.load_samples_file)
            replay.set_state(replay_state)
            logging.info(f"Loaded replay samples from file '{FLAGS.load_samples_file}'")
        except Exception:
            pass

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
            optimizer_state,
            scheduler_state,
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
    for i in range(FLAGS.num_actors):
        action_decoder = ContinousActionDecoder(action_embeddings.clone())
        net_config = {
            "action_encoder": ContinousActionEncoder(),
            "action_decoder": action_decoder,
            "action_space_dim": action_embeddings.shape[-1],
            "vit_config": VitConfig(),
            "num_planes": config.num_planes,
            "value_support_size": config.value_support_size,
            "reward_support_size": config.reward_support_size,
            "sequence_length": FLAGS.environment_frame_stack,
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
