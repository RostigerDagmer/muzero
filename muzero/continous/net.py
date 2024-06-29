import logging
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from muzero.util import normalize_hidden_state
from .represent import VitConfig, RepresentationViTGeneral, RepresentationLMPythia, RepresentationLMClip
from .io import ContinousEncoderHead
from muzero.network import DynamicsMLPNet, MuZeroNet, PredictionMLPNet

def initialize_weights(net: nn.Module) -> None:
    """Initialize weights for Conv2d and Linear layers using kaming initializer."""
    assert isinstance(net, nn.Module)

    for module in net.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if not module.weight.requires_grad:
                continue
            # nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            # nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')

            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        if isinstance(module, nn.MultiheadAttention):
            if module.in_proj_weight.requires_grad:
                nn.init.xavier_normal_(module.in_proj_weight)
            if module.out_proj.weight.requires_grad:
                nn.init.xavier_normal_(module.out_proj.weight)
            if module.in_proj_bias.requires_grad:
                nn.init.zeros_(module.in_proj_bias)


class ContinousDynamics(nn.Module):
    """Dynamics model for continuous action spaces."""

    """ Continous actions are encoded as embedding vectors from LLMs. """

    def __init__(
        self, hidden_dim: int, num_planes: int, support_size: int, action_space_dim: int, dropout: float = 0.3
    ):
        super().__init__()
        self.transition_net = nn.Sequential(
            nn.Linear(hidden_dim + action_space_dim, num_planes),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(num_planes, hidden_dim),
        )

        self.reward_net = nn.Sequential(
            nn.Linear(hidden_dim, num_planes),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(num_planes, support_size),
        )

    def forward(self, hidden_state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Given hidden_state state and encoded action, predict the state transition and reward."""

        assert hidden_state.shape[0] == action.shape[0]
        action = action.to(hidden_state.device)
        x = torch.cat([hidden_state, action], dim=1)

        hidden_state_ = self.transition_net(x)
        hidden_state_ += hidden_state # TODO: check if residual connection is necessary
        hidden_state_ = F.normalize(hidden_state_)

        reward_logits = self.reward_net(hidden_state_)
        # reward_logits = F.softmax(reward_logits, dim=1)
        return hidden_state, reward_logits
    
# class GoalDynamics(nn.Module):
    # TODO: this is incredibly difficult

class ContinousPrediction(nn.Module):
    """Prediction model for continuous action spaces."""

    """ Continous actions are encoded as embedding vectors from LLMs. """

    def __init__(self, hidden_dim: int, num_planes: int, support_size: int, action_space_dim: int):
        super().__init__()

        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim, num_planes),
            nn.ELU(),
            nn.Linear(num_planes, support_size),
        )

        self.policy_net = nn.Sequential(
            nn.Linear(hidden_dim, num_planes),
            nn.ELU(),
            nn.Linear(num_planes, action_space_dim),
        )

    def forward(self, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Given hidden_state state, predict the value, policy and reward."""

        value_logits = self.value_net(hidden_state)
        pi_logits = self.policy_net(hidden_state)
        # normalize the policy logits
        pi_logits = F.normalize(pi_logits, dim=-1)
        # normalize the value logits
        value_logits = F.softmax(value_logits, dim=1)
        return pi_logits, value_logits

class ContinousMuzeroNet(MuZeroNet):
    def __init__(
        self,
        action_encoder: Optional[nn.Module],
        action_decoder: Optional[nn.Module],
        action_space_dim: int,
        vit_config: VitConfig = VitConfig(),
        hidden_dim: int = 512,
        num_planes: int = 512,
        value_support_size: int = 31,
        reward_support_size: int = 31,
        sequence_length: int = 8,
        attention_heads: int = 8,
        encoder: ContinousEncoderHead = ContinousEncoderHead.clip,
    ):
        super().__init__(-1)
        if encoder == ContinousEncoderHead.clip:
            self.represent_net = RepresentationViTGeneral(
                hidden_dim=hidden_dim,
                num_planes=num_planes,
                seq_len=sequence_length,
                attention_heads=attention_heads,
                config=vit_config,
            )
        elif encoder == ContinousEncoderHead.pythia:
            self.represent_net = RepresentationLMPythia(
                hidden_dim=hidden_dim, num_planes=num_planes, seq_len=sequence_length, attention_heads=attention_heads
            )
        elif encoder == ContinousEncoderHead.clip_lm:
            self.represent_net = RepresentationLMClip(
                hidden_dim=hidden_dim,
                num_planes=num_planes,
                seq_len=sequence_length,
                attention_heads=attention_heads,
                config=vit_config,
            )

        if action_encoder and action_decoder:
            self.dynamics_net = ContinousDynamics(
                hidden_dim=hidden_dim,
                num_planes=num_planes,
                support_size=reward_support_size,
                action_space_dim=action_space_dim,
            )
            self.prediction_net = ContinousPrediction(
                hidden_dim=hidden_dim,
                num_planes=num_planes,
                support_size=reward_support_size,
                action_space_dim=action_space_dim,
            )
        else:
            # we use MLP for Dynamics and Prediction
            self.dynamics_net = DynamicsMLPNet(
                num_actions = action_space_dim,
                num_planes = num_planes,
                hidden_dim = hidden_dim,
                support_size = reward_support_size,
            )
            self.prediction_net = PredictionMLPNet(
                num_actions = action_space_dim,
                num_planes = num_planes,
                hidden_dim = hidden_dim,
                support_size = reward_support_size,
            )
        self.action_encoder = action_encoder
        self.action_decoder = action_decoder

        self.value_support_size = value_support_size
        self.reward_support_size = reward_support_size
        initialize_weights(self)

    def represent(self, x: torch.Tensor) -> torch.Tensor:
        hidden_state = self.represent_net(x)
        # print("hidden_state", hidden_state.shape)
        # if not self.action_decoder and self.action_encoder:
        #     hidden_state = normalize_hidden_state(hidden_state)
        return hidden_state

    def dynamics(self, hidden_state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_state, reward_logits = self.dynamics_net(hidden_state, action)
        # print("hidden_state", hidden_state.shape)
        # print("reward_logits", reward_logits.shape)
        if not self.action_decoder and self.action_encoder:
            logging.debug("Normalizing hidden state")
            hidden_state = normalize_hidden_state(hidden_state)
        return hidden_state, reward_logits

    def prediction(self, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pi_logits, value_logits = self.prediction_net(hidden_state)
        pi_logits = F.softmax(pi_logits, dim=-1)
        return pi_logits, value_logits


