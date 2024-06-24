import unittest
import torch

from muzero.continous.debug import plot_grad_flow
from muzero.continous.net import ContinousMuzeroNet, ContinousActionEncoder, ContinousActionDecoder
from muzero.continous.io import ContinousEncoderHead
from muzero.continous.represent import VitConfig

class TestContinousCLIPMuzeroNet(unittest.TestCase):

    def setUp(self):
        vit_config = VitConfig()
        action_space_dim = 512  # Assuming 2048 dimensions for the action space
        action_space_size = 16

        self.net = ContinousMuzeroNet(
            action_encoder=ContinousActionEncoder(),
            action_decoder=ContinousActionDecoder(action_set=torch.randn(action_space_size, action_space_dim).to("mps")),  # Dummy action set
            action_space_dim=action_space_dim,
            vit_config=vit_config,
            num_planes=512,
            reward_support_size=31,
            sequence_length=8,
            attention_heads=8,
            encoder=ContinousEncoderHead.clip
        ).to("mps")
        total_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        print("Total parameters:", total_params)
        self.input_shape = (3 * 8, 224, 224)  # Example input shape for an image
        self.batch_size = 2
        self.action_space_dim = action_space_dim
        self.action_space_size = action_space_size

    def test_forward_pass(self):
        # Simulate a batch of images
        input_data = torch.randn(self.batch_size, *self.input_shape)
        # Simulate a batch of actions
        actions = torch.randn(self.batch_size, self.action_space_dim)

        # Perform the forward pass
        hidden_state = self.net.represent(input_data)
        next_hidden_state, reward_logits = self.net.dynamics(hidden_state, actions)
        policy_logits, value_logits = self.net.prediction(hidden_state)

        # Check if computations are successful (no exceptions raised)
        self.assertTrue(hidden_state is not None)
        self.assertTrue(next_hidden_state is not None)
        self.assertTrue(reward_logits is not None)
        self.assertTrue(policy_logits is not None)
        self.assertTrue(value_logits is not None)


    def test_output_shapes(self):
        input_data = torch.randn(self.batch_size, *self.input_shape)
        actions = torch.randn(self.batch_size, self.action_space_dim)

        hidden_state = self.net.represent(input_data)
        next_hidden_state, reward_logits = self.net.dynamics(hidden_state, actions)
        policy_logits, value_logits = self.net.prediction(hidden_state)

        self.assertEqual(hidden_state.shape, torch.Size([self.batch_size, 512]))  # Check the hidden state shape
        self.assertEqual(next_hidden_state.shape, torch.Size([self.batch_size, 512]))  # Check the next hidden state shape
        self.assertEqual(reward_logits.shape, torch.Size([self.batch_size, 31]))  # Assuming 31 is the support size for rewards
        self.assertEqual(policy_logits.shape, torch.Size([self.batch_size, self.action_space_dim]))  # Check the policy logits shape
        self.assertEqual(value_logits.shape, torch.Size([self.batch_size, 31]))  # Assuming 31 is the support size for values

    def test_weight_freezing(self):
        from PIL import Image
        # Check weights before forward pass
        initial_encoder_weight = next(self.net.represent_net.encoder.parameters()).clone()

        # Perform forward and backward passes
        input_data = torch.randn(self.batch_size, *self.input_shape)
        actions = torch.randn(self.batch_size, self.action_space_dim)
        hidden_state = self.net.represent(input_data)
        next_hidden_state, reward_logits = self.net.dynamics(hidden_state, actions)
        policy_logits, value_logits = self.net.prediction(hidden_state)

        print("policy_logits", policy_logits)
        print("value_logits", value_logits)
        # Compute loss and perform backpropagation
        loss = reward_logits.sum() + policy_logits.sum() + value_logits.sum()
        print("loss", loss)
        loss.backward()

        # Check weights after backward pass
        post_encoder_weight = next(self.net.represent_net.encoder.parameters())

        # Ensure encoder weights have not changed
        self.assertTrue(torch.equal(initial_encoder_weight, post_encoder_weight))

    def calc_loss(self):
        from muzero.pipeline import calc_loss
        from muzero.replay import Transition
        import numpy as np

        print("testing calc_loss")
        num_actions = 5
        # simulate transitions

        state=np.random.randn(self.batch_size, *self.input_shape) * 255
        action=np.random.randn(self.batch_size, num_actions, self.action_space_dim)
        action = action / np.linalg.norm(action, axis=-1, keepdims=True)
        pi_prob=np.random.standard_normal(size=(self.batch_size, num_actions, self.action_space_dim))
        pi_prob = pi_prob / np.linalg.norm(pi_prob, axis=-1, keepdims=True)
        value=np.random.randn(self.batch_size, num_actions)
        reward=np.random.randn(self.batch_size, num_actions)

        transition = Transition(
            state,
            action,
            pi_prob,
            value,
            reward,
        )

        weights = torch.randn(self.batch_size, 1)
        l, _ = calc_loss(self.net, "mps", transition, weights)
        l.backward()
        # plot gradient flow
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)
        plot_grad_flow(self.net)

    def self_play(self):
        from muzero.pipeline import run_self_play
        from muzero.config import make_atari_config
        from muzero.gym_env import create_atari_environment
        from muzero.atari_v2.run_training import ActionEncoderWith
        import multiprocessing as mp
        import numpy as np

        random_state = np.random.RandomState(0)

        def environment_builder():
            return create_atari_environment(
                env_name="Breakout",
                screen_height=224,
                screen_width=224,
                frame_skip=4,
                frame_stack=8,
                max_episode_steps=10,
                seed=random_state.randint(1, 2**31),
                noop_max=30,
                terminal_on_life_loss=False,
                clip_reward=False,
                output_actions=True,
                resize_and_gray=False
            )

        self_play_envs, self_play_actions = zip(*[environment_builder() for _ in range(1)])

        eval_env, eval_actions = environment_builder()

        input_shape = self_play_envs[0].observation_space.shape
        num_actions = self_play_envs[0].action_space.n

        tag = self_play_envs[0].spec.id

        config = make_atari_config(
            num_training_steps=100,
            batch_size=32,
            min_replay_size=10,
            use_tensorboard=False,
            clip_grad=True,
        )

        formatted_actions = [f"action: {action}" for action in eval_actions]
        print(f"formatted actions: {formatted_actions}")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenized_actions = tokenizer(formatted_actions, padding=True, return_tensors="pt")
        # print(f"tokenized actions: {tokenized_actions}")
        action_encoder = ContinousActionEncoder()
        action_embeddings = action_encoder(tokenized_actions.input_ids, tokenized_actions.attention_mask)
        action_decoder = ContinousActionDecoder(action_embeddings.clone())
        net_config = {
            "action_encoder": ContinousActionEncoder(),
            "action_decoder": action_decoder,
            "action_space_dim": action_embeddings.shape[-1],
            "vit_config": VitConfig(),
            "num_planes": config.num_planes,
            "value_support_size": config.value_support_size,
            "reward_support_size": config.reward_support_size,
            "sequence_length": 8,
            "attention_heads": 8, # attention heads
        }
        data_queue = mp.Queue()
        train_steps_counter = mp.Value('i', 0)
        stop_event = mp.Event()

        action_encoder = ActionEncoderWith(action_embeddings.clone())
        run_self_play(
                config,
                0,
                net_config,
                "mps",
                self_play_envs[0],
                data_queue,
                train_steps_counter,
                stop_event,
                tag,
                True,
                action_decoder,
                action_encoder,)