import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Any, Callable, TypeVar
from abc import abstractmethod
from transformers import GPTNeoXForCausalLM, AutoTokenizer, BatchEncoding
import open_clip
from .positional_encoding import RotaryPositionalEncoding
from torchvision.transforms.v2 import Compose, Resize, Normalize, ToDtype, ToImage
import logging

class VitConfig:
    def __init__(self, base_model: str = 'ViT-B-32', pretrained: str = 'laion2b_s34b_b79k'):
        self.base_model = base_model
        self.pretrained = pretrained


mean = (0.48145466, 0.4578275, 0.40821073)
std = (0.26862954, 0.26130258, 0.27577711)


class AddGaussianNoise(object):
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.std = std
        self.mean = mean

    def __call__(self, tensor: torch.Tensor):
        return tensor + torch.randn(tensor.size(), device=tensor.device) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class RepresentationGeneral(nn.Module):
    T = TypeVar('T')

    def __init__(
        self,
        encoder: Callable[[T], torch.Tensor],
        embedding_dim: int,
        state_space_dim: int,
        num_planes: int,
        seq_len: int,
        attention_heads: int,
        transformer_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.state_space_dim = state_space_dim
        self.num_planes = num_planes
        self.seq_len = seq_len
        self.encoder = encoder
        self.transformer_layer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=attention_heads),
            num_layers=transformer_layers,
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, num_planes),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_planes, embedding_dim),
        )
        self.positional_encoding = RotaryPositionalEncoding(self.embedding_dim)

    @abstractmethod
    def preprocess(self, x: Any) -> T:
        pass

    def forward(self, x: Any) -> torch.Tensor:
        # print("x.shape in forward: ", x.shape)
        # print("x in forward: ", x)

        x = self.preprocess(x)
        e = self.encoder(x)
        x = e.reshape(-1, self.seq_len, self.embedding_dim)
        x = self.positional_encoding(x)
        y = self.transformer_layer(x)
        y = self.pool(y.permute(0, 2, 1))
        z = self.mlp(y.view(-1, self.embedding_dim))
        return F.normalize(z)


class RepresentationViTGeneral(RepresentationGeneral):
    def __init__(
        self,
        hidden_dim: int,
        num_planes: int,
        seq_len: int,
        attention_heads: int,
        config: VitConfig,
        transformer_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__(None, 512, hidden_dim, num_planes, seq_len, attention_heads, transformer_layers, dropout)
        print("hidden_dim: ", hidden_dim)
        self.clip, _, _ = open_clip.create_model_and_transforms(config.base_model, pretrained=config.pretrained)
        self.transform = Compose(
            [
                ToImage(),
                Resize((224, 224), antialias=False),
                AddGaussianNoise(0.0, 0.01),
                Normalize(mean=mean, std=std),
                ToDtype(torch.float32, scale=True)
            ]
        )
        for param in self.clip.parameters():
            param.requires_grad = False

        self.encoder: Callable[[torch.Tensor], torch.Tensor] = lambda x: self.clip.encode_image(x, normalize=True)
        self.transformer_layer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=attention_heads),
            num_layers=transformer_layers,
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, num_planes),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_planes, hidden_dim),
        )
        self.positional_encoding = RotaryPositionalEncoding(self.embedding_dim)
        # self.layernorm = nn.LayerNorm(hidden_dim)
        self.seq_len = seq_len

    def preprocess(self, x: Any) -> torch.Tensor:
        B, C, H, W = x.shape
        assert C // self.seq_len == 3
        x = x.view(B * self.seq_len, 3, H, W)
        return self.transform(x)


def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


class RepresentationLMPythia(RepresentationGeneral):
    def __init__(
        self,
        hidden_dim: int,
        num_planes: int,
        seq_len: int,
        attention_heads: int,
        transformer_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__(None, 512, hidden_dim, num_planes, seq_len, attention_heads, transformer_layers, dropout)
        self.model = GPTNeoXForCausalLM.from_pretrained(
            "EleutherAI/pythia-70m-deduped",
            revision="step3000",
            device_map="cpu",
        )
        if not isinstance(self.model, GPTNeoXForCausalLM):
            raise ValueError("Model is not GPTNeoXForCausalLM")

        self.tokenizer = AutoTokenizer.from_pretrained(
            "EleutherAI/pythia-70m-deduped",
            revision="step3000",
            device_map="cpu",
        )
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        for param in self.model.parameters():
            param.requires_grad = False
        self.encoder = self.encode

    def encode(self, x: BatchEncoding) -> torch.Tensor:
        out = self.model(**x.to(self.model.device), output_hidden_states=True, return_dict=True)
        last_hidden_state = out.hidden_states[-1]
        return last_token_pool(last_hidden_state, x['attention_mask'])

    def preprocess(self, x: torch.Tensor) -> BatchEncoding:
        def format(elem: torch.Tensor):
            logging.debug(f"elem: {elem}")
            return f"cart position: {elem[0]}; cart velocity: {elem[1]}; pole angle: {elem[2]}; pole angular velocity: {elem[3]}"

        x: list[str] = [format(seq_elem) for seq in x for seq_elem in seq]
        return self.tokenizer(x, padding=True, return_tensors="pt")


class RepresentationLMClip(RepresentationGeneral):
    def __init__(
        self,
        hidden_dim: int,
        num_planes: int,
        seq_len: int,
        attention_heads: int,
        config: VitConfig = VitConfig(),
        transformer_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__(None, 512, hidden_dim, num_planes, seq_len, attention_heads, transformer_layers, dropout)
        self.model, _, _ = open_clip.create_model_and_transforms(config.base_model, config.pretrained)
        for param in self.model.parameters():
            param.requires_grad = False

        self.tokenizer = open_clip.get_tokenizer(config.base_model)
        self.encoder = lambda x: self.model.encode_text(x.to(next(self.model.buffers()).device), normalize=True)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:

        def format(elem: torch.Tensor) -> str:
            print(f"elem: {elem}")
            return f"cart position: {elem[0]}; cart velocity: {elem[1]}; pole angle: {elem[2]}; pole angular velocity: {elem[3]}"

        y: list[str] = [format(seq_elem) for seq in x for seq_elem in seq]
        return self.tokenizer(y)