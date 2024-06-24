import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_length: int):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor):
        return x + self.pe[:, : x.size(1)]


class RotaryPositionalEncoding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor):
        d = x.shape[-1]
        pos = torch.arange(x.shape[1]).unsqueeze(-1)
        half_dim = d // 2
        theta = pos / torch.pow(10000, (2 * (torch.arange(half_dim)) / d))

        sin_theta = torch.sin(theta).to(x.device)
        cos_theta = torch.cos(theta).to(x.device)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        x_rotated_even = cos_theta * x_even - sin_theta * x_odd
        x_rotated_odd = sin_theta * x_even + cos_theta * x_odd

        x_rotated = torch.empty_like(x)
        x_rotated[..., 0::2] = x_rotated_even
        x_rotated[..., 1::2] = x_rotated_odd

        return x_rotated