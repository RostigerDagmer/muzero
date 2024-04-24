from enum import Enum
from typing import Tuple
import torch
import torch.nn as nn
import open_clip
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextStreamer

from muzero.network import MuZeroNet
from muzero.util import normalize_hidden_state

class VitConfig:
    def __init__(self, base_model: str = 'ViT-B-32', pretrained: str = 'laion2b_s34b_b79k'):
        self.base_model = base_model
        self.pretrained = pretrained


class RepresentationViTGeneral(nn.Module):
    def __init__(self, hidden_dim: int, num_planes: int, config: VitConfig):
        super().__init__()
        input_size = 512 # feature vector length
        self.encoder, _, self.preprocess = open_clip.create_model_and_transforms(config.base_model, pretrained=config.pretrained)
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.mlp = nn.Sequential(
            nn.Linear(input_size, num_planes),
            nn.ReLU(),
            nn.Linear(num_planes, hidden_dim),
        )

    def forward(self, x):
        x = self.preprocess(x) # transform image input
        e = self.encoder(x) # embed image input
        return self.mlp(e) # project to hidden dim
    

class ContinousDynamics(nn.Module):
    """ Dynamics model for continuous action spaces."""
    """ Continous actions are encoded as embedding vectors from LLMs. """

    def __init__(self, hidden_dim: int, num_planes: int, support_size: int, action_space_dim: int):
        super().__init__()
        self.transition_net = nn.Sequential(
            nn.Linear(hidden_dim + action_space_dim, num_planes),
            nn.ReLU(),
            nn.Linear(num_planes, hidden_dim),
        )

        self.reward_net = nn.Sequential(
            nn.Linear(hidden_dim, num_planes),
            nn.ReLU(),
            nn.Linear(num_planes, support_size),
        )

    def forward(self, hidden_state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Given hidden_state state and encoded action, predict the state transition and reward."""

        assert hidden_state.shape[0] == action.shape[0]

        # [batch_size, num_actions]
        x = torch.cat([hidden_state, action], dim=1)

        hidden_state = self.transition_net(x)
        reward_logits = self.reward_net(hidden_state)
        return hidden_state, reward_logits

class ContinousPrediction(nn.Module):
    """ Prediction model for continuous action spaces."""
    """ Continous actions are encoded as embedding vectors from LLMs. """

    def __init__(self, hidden_dim: int, num_planes: int, support_size: int, action_space_dim: int):
        super().__init__()

        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim, num_planes),
            nn.ReLU(),
            nn.Linear(num_planes, support_size),
        )

        self.policy_net = nn.Sequential(
            nn.Linear(hidden_dim, num_planes),
            nn.ReLU(),
            nn.Linear(num_planes, action_space_dim),
        )

    def forward(self, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Given hidden_state state, predict the value, policy and reward."""

        value_logits = self.value_net(hidden_state)
        policy_logits = self.policy_net(hidden_state)
        return policy_logits, value_logits
    


tokenizer = AutoTokenizer.from_pretrained("NousResearch/OLMo-Bitnet-1B")

class EmbeddingMethod(Enum):
    mean = 1
    max = 2
    combined = 3
    last_token_pool = 4

class ContinousActionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.action_encoder = AutoModelForCausalLM.from_pretrained("NousResearch/OLMo-Bitnet-1B", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
        for param in self.action_encoder.parameters():
            param.requires_grad = False

    def last_token_pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, method: EmbeddingMethod) -> torch.Tensor:
        out = self.action_encoder.forward(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
        last_hidden_state = out.hidden_states[-1]
        if method == EmbeddingMethod.mean:
            action = last_hidden_state.mean(dim=1)
        elif method == EmbeddingMethod.max:
            action, _ = last_hidden_state.max(dim=1)
        elif method == EmbeddingMethod.combined:
            mean_pooled = last_hidden_state.mean(dim=1)
            max_pooled, _ = last_hidden_state.max(dim=1)
            action = torch.cat((mean_pooled, max_pooled), dim=1)
        elif method == EmbeddingMethod.last_token_pool:
            action = self.last_token_pool(last_hidden_state, attention_mask)
        return action



class ContinousActionDecoder(nn.Module):
    def __init__(self, action_set: torch.Tensor):
        """ Action set is a set of preembedded actions that have to be searched for the closest action. """
        """ The advantage of encoding actions is that properties and layout of a specific action space is not baked into the model architecture. Only the encoder. """
        super().__init__()
        self.action_set = action_set

    def forward(self, pred_action: torch.Tensor) -> torch.Tensor:
        """ Given the predicted action, find the closest action in the action set. """
        return self.action_set[torch.cdist(pred_action, self.action_set).argmin(dim=1)]
    

class ContinousMuzeroNet(MuZeroNet):
    def __init__(
            self,
            action_encoder: nn.Module,
            action_decoder: nn.Module,
            action_space_dim: int,
            vit_config: VitConfig,
            num_planes: int = 512,
            reward_support_size: int = 31,
        ):
        super().__init__(-1)
        self.represent_net = RepresentationViTGeneral(hidden_dim=num_planes, num_planes=num_planes, config=vit_config)
        self.dynamics_net = ContinousDynamics(hidden_dim=num_planes, num_planes=num_planes, support_size=reward_support_size, action_space_dim=action_space_dim)
        self.prediction_net = ContinousPrediction(hidden_dim=num_planes, num_planes=num_planes, support_size=reward_support_size, action_space_dim=action_space_dim)
        self.action_encoder = action_encoder
        self.action_decoder = action_decoder
    
    def represent(self, x: torch.Tensor) -> torch.Tensor:
        hidden_state = self.represent_net(x)
        hidden_state = normalize_hidden_state(hidden_state)
        return hidden_state

    def dynamics(self, hidden_state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_state, reward_logits = self.dynamics_net(hidden_state, action)
        hidden_state = normalize_hidden_state(hidden_state)
        return hidden_state, reward_logits
    
    def prediction(self, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.prediction_net(hidden_state)
    

import unittest

class TestContinousMuzeroNet(unittest.TestCase):
    
    def setUp(self):
        vit_config = VitConfig()
        action_space_dim = 2048  # Assuming 2048 dimensions for the action space
        self.net = ContinousMuzeroNet(
            action_encoder=ContinousActionEncoder(),
            action_decoder=ContinousActionDecoder(action_set=torch.randn(10, action_space_dim)),  # Dummy action set
            action_space_dim=action_space_dim,
            vit_config=vit_config,
            num_planes=512,
            reward_support_size=31,
        )
        self.input_shape = (3, 224, 224)  # Example input shape for an image
        self.batch_size = 2

    def test_forward_pass(self):
        # Simulate a batch of images
        input_data = torch.randn(self.batch_size, *self.input_shape)
        # Simulate a batch of actions
        actions = torch.randn(self.batch_size, 2048)

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
        actions = torch.randn(self.batch_size, 2048)

        hidden_state = self.net.represent(input_data)
        next_hidden_state, reward_logits = self.net.dynamics(hidden_state, actions)
        policy_logits, value_logits = self.net.prediction(hidden_state)

        self.assertEqual(hidden_state.shape, torch.Size([self.batch_size, 512]))  # Check the hidden state shape
        self.assertEqual(next_hidden_state.shape, torch.Size([self.batch_size, 512]))  # Check the next hidden state shape
        self.assertEqual(reward_logits.shape, torch.Size([self.batch_size, 31]))  # Assuming 31 is the support size for rewards
        self.assertEqual(policy_logits.shape, torch.Size([self.batch_size, 2048]))  # Check the policy logits shape
        self.assertEqual(value_logits.shape, torch.Size([self.batch_size, 31]))  # Assuming 31 is the support size for values

    def test_weight_freezing(self):
        # Check weights before forward pass
        initial_encoder_weight = next(self.net.represent_net.encoder.parameters()).clone()

        # Perform forward and backward passes
        input_data = torch.randn(self.batch_size, *self.input_shape)
        actions = torch.randn(self.batch_size, 2048)
        hidden_state = self.net.represent(input_data)
        next_hidden_state, reward_logits = self.net.dynamics(hidden_state, actions)
        policy_logits, value_logits = self.net.prediction(hidden_state)

        # Compute loss and perform backpropagation
        loss = reward_logits.sum() + policy_logits.sum() + value_logits.sum()
        loss.backward()

        # Check weights after backward pass
        post_encoder_weight = next(self.net.represent_net.encoder.parameters())
        
        # Ensure encoder weights have not changed
        self.assertTrue(torch.equal(initial_encoder_weight, post_encoder_weight))


if __name__ == '__main__':
    unittest.main()


