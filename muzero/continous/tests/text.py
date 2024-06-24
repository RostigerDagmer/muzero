import logging
import unittest
import torch

from muzero.continous.debug import NoMatplotFilter, plot_grad_flow
from muzero.continous.net import ContinousMuzeroNet
from muzero.continous.io import ContinousEncoderHead, ContinousActionEncoder, ContinousActionDecoder
from muzero.continous.represent import VitConfig

logging.basicConfig(level=logging.DEBUG)
for handler in logging.root.handlers:
    handler.addFilter(NoMatplotFilter())

class TestContinousPythiaMuerzoNet(unittest.TestCase):

    def setUp(self):
        vit_config = VitConfig()
        action_space_dim = 1024  # Assuming 2048 dimensions for the action space
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
            encoder=ContinousEncoderHead.pythia
        ).to("mps")
        total_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        print("Total parameters: ", total_params)
        self.batch_size = 2
        self.action_space_dim = action_space_dim
        self.action_space_size = action_space_size

    def test_forward_pass(self):
        # Simulate a batch of images
        input_data = [["pos: 0.12512612, 12.121950761, vel: 0.0, 0.0"] * self.batch_size] * self.net.represent_net.seq_len
        # Simulate a batch of actions
        actions = torch.randn(self.batch_size, self.action_space_dim)

        # Perform the forward pass
        hidden_state = self.net.represent(input_data)
        print("hidden_state.shape: ", hidden_state.shape)
        next_hidden_state, reward_logits = self.net.dynamics(hidden_state, actions)
        policy_logits, value_logits = self.net.prediction(hidden_state)

        # Check if computations are successful (no exceptions raised)
        self.assertTrue(hidden_state is not None)
        self.assertTrue(next_hidden_state is not None)
        self.assertTrue(reward_logits is not None)
        self.assertTrue(policy_logits is not None)
        self.assertTrue(value_logits is not None)


class TestContinousClipLMMuerzoNet(unittest.TestCase):

    def setUp(self):
        vit_config = VitConfig()
        self.action_space_dim = 1024  # Assuming 2048 dimensions for the action space
        self.action_space_size = 16
        self.input_shape = (8, 4)

        self.net = ContinousMuzeroNet(
            action_encoder=ContinousActionEncoder(vit_config=vit_config),
            action_decoder=ContinousActionDecoder(
                action_set=torch.randn(self.action_space_size, self.action_space_dim).to("mps")
            ),  # Dummy action set
            action_space_dim=self.action_space_dim,
            vit_config=vit_config,
            num_planes=512,
            reward_support_size=31,
            sequence_length=8,
            attention_heads=8,
            encoder=ContinousEncoderHead.clip_lm,
        ).to("mps")
        total_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        print("Total parameters:", total_params)
        self.batch_size = 2

    def test_forward_pass(self):
        # Simulate a batch of images
        input_data = torch.rand(self.batch_size, self.net.represent_net.seq_len, 4)
        # Simulate a batch of actions
        actions = torch.randn(self.batch_size, self.action_space_dim)

        # Perform the forward pass
        hidden_state = self.net.represent(input_data.to('mps'))
        print("hidden_state.shape: ", hidden_state.shape)
        next_hidden_state, reward_logits = self.net.dynamics(hidden_state, actions)
        policy_logits, value_logits = self.net.prediction(hidden_state)

        # Check if computations are successful (no exceptions raised)
        self.assertTrue(hidden_state is not None)
        self.assertTrue(next_hidden_state is not None)
        self.assertTrue(reward_logits is not None)
        self.assertTrue(policy_logits is not None)
        self.assertTrue(value_logits is not None)

    def test_calc_loss(self):
        from muzero.pipeline import calc_loss
        from muzero.replay import Transition
        import numpy as np

        print("testing calc_loss")
        num_actions = 2
        # simulate transitions

        state = np.random.rand(self.batch_size, *self.input_shape)
        print(f"state.shape: {state.shape}")
        action = np.random.rand(self.batch_size, num_actions, self.action_space_dim)
        action /= np.sqrt(np.sum(action * action, axis=-1))[:, :, np.newaxis]
        pi_prob = np.random.standard_normal(size=(self.batch_size, num_actions, self.action_space_dim))
        pi_prob /= np.sqrt(np.sum(pi_prob * pi_prob, axis=-1))[:, :, np.newaxis]
        value = np.random.rand(self.batch_size, num_actions)
        reward = np.random.rand(self.batch_size, num_actions)

        transition = Transition(
            state,
            action,
            pi_prob,
            value,
            reward,
        )

        weights = torch.ones(self.batch_size, 1)
        l, _ = calc_loss(self.net, torch.device("mps"), transition, weights)
        l.backward()
        # plot gradient flow
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)
        plot_grad_flow(self.net)

    
if __name__ == "__main__":
    unittest.main()