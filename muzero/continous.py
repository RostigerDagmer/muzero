from enum import Enum
import math
from typing import Tuple
import torch
import torch.nn as nn
import open_clip
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextStreamer
from torchvision.transforms.v2 import Compose, Resize, Normalize, ToTensor, ToImage
import torch.nn.functional as F

from muzero.network import MuZeroNet
from muzero.util import normalize_hidden_state
import os

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        # print("positional encoding -> d_model: ", d_model)
        # print("positional encoding -> max_seq_length: ", max_seq_length)
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        # print("pe shape: ", self.pe.shape)
        # print("positional encoding -> x shape: ", x.shape)
        return x + self.pe[:, :x.size(1)]

class VitConfig:
    def __init__(self, base_model: str = 'ViT-B-32', pretrained: str = 'laion2b_s34b_b79k'):
        self.base_model = base_model
        self.pretrained = pretrained

mean = (0.48145466, 0.4578275, 0.40821073)
std = (0.26862954, 0.26130258, 0.27577711)

class RepresentationViTGeneral(nn.Module):
    def __init__(self, hidden_dim: int, num_planes: int, seq_len: int, attention_heads: int, config: VitConfig):
        super().__init__()
        input_size = 512 # feature vector length
        print("hidden_dim: ", hidden_dim)
        self.encoder, _, _ = open_clip.create_model_and_transforms(config.base_model, pretrained=config.pretrained)
        self.preprocess = Compose([
            ToImage(),
            Resize((224, 224), antialias=True),
            Normalize(mean=mean, std=std),
            ToTensor()
        ])
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.mlp = nn.Sequential(
            nn.Linear(input_size * seq_len, num_planes),
            nn.ReLU(),
            nn.Linear(num_planes, hidden_dim),
        )
        self.attention_block = nn.MultiheadAttention(hidden_dim, attention_heads)
        self.positional_encoding = PositionalEncoding(input_size, seq_len)
        self.layernorm = nn.LayerNorm(hidden_dim)
        self.seq_len = seq_len

    def forward(self, x):
        B, C, H, W = x.shape
        assert C // self.seq_len == 3
        # print("represent input shape: ", x.shape)
        # unfold Channel dimension to seq_len (B * seq_len, 3, H, W) images
        x = x.view(B * self.seq_len, 3, H, W)
        
        # print("unfolded shape: ", x.shape)
        x = self.preprocess(x) # transform image input
        e = self.encoder.encode_image(x) # embed image input
        # print("e shape: ", e.shape)
        # x shape is B * seq_len, 512 e.g. 2 * 8, 512
        # print("e", e.shape)
        # recover the original batch size
        x = e.view(B, self.seq_len, 512)
        residual = x
        # print("x", x.shape)
        # debug_plot(x.clone())
        # print("prepositional shape: ", x.shape)
        x = self.positional_encoding(x)
        # debug_plot(x.clone())
        y = self.attention_block(e, e, e, need_weights=False)[0]
        y = self.layernorm(y)
        y = y.view(B, -1) # flatten the sequence
        z = self.mlp(y)
        print("z shape: ", z.shape)
        print("residual shape: ", residual.shape)
        z = z + residual
        z = F.relu(z)
        return z # project to hidden dim
    

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
        if action.shape[0] != 1: 
            print("action shape in dynamics: ", action.shape)
        # [batch_size, num_actions]
        residual = hidden_state
        x = torch.cat([hidden_state, action.squeeze(1)], dim=1)

        hidden_state = self.transition_net(x)
        hidden_state = hidden_state + residual
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
        pi_logits = self.policy_net(hidden_state)
        return pi_logits, value_logits
    

def debug_plot(x: torch.Tensor):
    import matplotlib.pyplot as plt
    if x.dim() == 4:
        print("DEBUG: 4D tensor, taking first entry in the batch.")
        x = x[0]
    if x.shape[0] > 3:
        print("DEBUG: more than 3 channels, taking the first channel.")
        x = x[0]
    elif x.shape[0] == 2:
        print("DEBUG: 2 channels, taking the first channel.")
        x = x[0]
    plt.imshow(x)
    plt.show()

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


    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, method: EmbeddingMethod = EmbeddingMethod.max) -> torch.Tensor:
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
        return action.float()



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
            vit_config: VitConfig = VitConfig(),
            num_planes: int = 512,
            value_support_size: int = 31,
            reward_support_size: int = 31,
            sequence_length: int = 8,
            attention_heads: int = 8
        ):
        super().__init__(-1)
        self.represent_net = RepresentationViTGeneral(hidden_dim=num_planes, num_planes=num_planes, seq_len=sequence_length, attention_heads=attention_heads, config=vit_config)
        self.dynamics_net = ContinousDynamics(hidden_dim=num_planes, num_planes=num_planes, support_size=reward_support_size, action_space_dim=action_space_dim)
        self.prediction_net = ContinousPrediction(hidden_dim=num_planes, num_planes=num_planes, support_size=reward_support_size, action_space_dim=action_space_dim)
        self.action_encoder = action_encoder
        self.action_decoder = action_decoder
        self.value_support_size = value_support_size
        self.reward_support_size = reward_support_size
    
    def represent(self, x: torch.Tensor) -> torch.Tensor:
        hidden_state = self.represent_net(x)
        # print("hidden_state", hidden_state.shape)
        hidden_state = normalize_hidden_state(hidden_state)
        return hidden_state

    def dynamics(self, hidden_state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_state, reward_logits = self.dynamics_net(hidden_state, action)
        # print("hidden_state", hidden_state.shape)
        # print("reward_logits", reward_logits.shape)
        hidden_state = normalize_hidden_state(hidden_state)
        return hidden_state, reward_logits
    
    def prediction(self, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.prediction_net(hidden_state)


def plot_grad_flow(named_parameters):
    import matplotlib.pyplot as plt
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)

import unittest

class TestContinousMuzeroNet(unittest.TestCase):
    
    def setUp(self):
        vit_config = VitConfig()
        action_space_dim = 2048  # Assuming 2048 dimensions for the action space
        action_space_size = 16
        
        self.net = ContinousMuzeroNet(
            action_encoder=ContinousActionEncoder(),
            action_decoder=ContinousActionDecoder(action_set=torch.randn(10, action_space_dim)),  # Dummy action set
            action_space_dim=action_space_dim,
            vit_config=vit_config,
            num_planes=512,
            reward_support_size=31,
            sequence_length=8,
            attention_heads=8
        ).to("cuda")
        total_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        print("Total parameters:", total_params)
        self.input_shape = (3 * 8, 224, 224)  # Example input shape for an image
        self.batch_size = 2
        self.action_space_dim = action_space_dim
        self.action_space_size = action_space_size

    # def test_forward_pass(self):
    #     # Simulate a batch of images
    #     input_data = torch.randn(self.batch_size, *self.input_shape)
    #     # Simulate a batch of actions
    #     actions = torch.randn(self.batch_size, self.action_space_dim)

    #     # Perform the forward pass
    #     hidden_state = self.net.represent(input_data)
    #     next_hidden_state, reward_logits = self.net.dynamics(hidden_state, actions)
    #     policy_logits, value_logits = self.net.prediction(hidden_state)
        
    #     # Check if computations are successful (no exceptions raised)
    #     self.assertTrue(hidden_state is not None)
    #     self.assertTrue(next_hidden_state is not None)
    #     self.assertTrue(reward_logits is not None)
    #     self.assertTrue(policy_logits is not None)
    #     self.assertTrue(value_logits is not None)
        

    # def test_output_shapes(self):
    #     input_data = torch.randn(self.batch_size, *self.input_shape)
    #     actions = torch.randn(self.batch_size, self.action_space_dim)

    #     hidden_state = self.net.represent(input_data)
    #     next_hidden_state, reward_logits = self.net.dynamics(hidden_state, actions)
    #     policy_logits, value_logits = self.net.prediction(hidden_state)

    #     self.assertEqual(hidden_state.shape, torch.Size([self.batch_size, 512]))  # Check the hidden state shape
    #     self.assertEqual(next_hidden_state.shape, torch.Size([self.batch_size, 512]))  # Check the next hidden state shape
    #     self.assertEqual(reward_logits.shape, torch.Size([self.batch_size, 31]))  # Assuming 31 is the support size for rewards
    #     self.assertEqual(policy_logits.shape, torch.Size([self.batch_size, self.action_space_dim]))  # Check the policy logits shape
    #     self.assertEqual(value_logits.shape, torch.Size([self.batch_size, 31]))  # Assuming 31 is the support size for values

    # def test_weight_freezing(self):
    #     from PIL import Image
    #     # Check weights before forward pass
    #     initial_encoder_weight = next(self.net.represent_net.encoder.parameters()).clone()

    #     # Perform forward and backward passes
    #     input_data = torch.randn(self.batch_size, *self.input_shape)
    #     actions = torch.randn(self.batch_size, self.action_space_dim)
    #     hidden_state = self.net.represent(input_data)
    #     next_hidden_state, reward_logits = self.net.dynamics(hidden_state, actions)
    #     policy_logits, value_logits = self.net.prediction(hidden_state)

    #     print("policy_logits", policy_logits)
    #     print("value_logits", value_logits)
    #     # Compute loss and perform backpropagation
    #     loss = reward_logits.sum() + policy_logits.sum() + value_logits.sum()
    #     print("loss", loss)
    #     loss.backward()

    #     # Check weights after backward pass
    #     post_encoder_weight = next(self.net.represent_net.encoder.parameters())
        
    #     # Ensure encoder weights have not changed
    #     self.assertTrue(torch.equal(initial_encoder_weight, post_encoder_weight))

    def test_calc_loss(self):
        from muzero.pipeline import calc_loss
        from muzero.replay import Transition
        import numpy as np
        
        print("testing calc_loss")
        # simulate transitions
        transition = Transition(
            state=np.random.randn(self.batch_size, *self.input_shape),
            action=np.random.randn(self.batch_size, 5, self.action_space_dim),
            pi_prob=np.random.randn(self.batch_size, self.action_space_size),
            value=np.random.randn(self.batch_size, 1),
            reward=np.random.randn(self.batch_size, 1)
        )
        
        weights = torch.randn(self.batch_size, 1)
        l, _ = calc_loss(self.net, "cuda", transition, weights)
        l.backward()
        # plot gradient flow
        plot_grad_flow(self.net.named_parameters())
        

if __name__ == '__main__':
    unittest.main()


