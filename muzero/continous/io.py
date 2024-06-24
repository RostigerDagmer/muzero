import logging
import torch
import torch.nn as nn
from enum import Enum
from transformers import GPTNeoXForCausalLM
import open_clip
from typing import Optional, Any
from .represent import VitConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from .represent import last_token_pool


# tokenizer = AutoTokenizer.from_pretrained(
#   "EleutherAI/pythia-70m-deduped",
#   revision="step3000",
#   device_map="auto"
# )

# tokenizer = open_clip.get_tokenizer('ViT-B-32')

class ContinousEncoderHead(Enum):
    pythia = 1
    clip = 2
    clip_lm = 3

class EmbeddingMethod(Enum):
    mean = 1
    max = 2
    combined = 3
    last_token_pool = 4


class ContinousActionEncoder(nn.Module):
    def __init__(self, vit_config: Optional[VitConfig] = None, device: str = "auto"):
        super().__init__()
        if vit_config is None:
            model = GPTNeoXForCausalLM.from_pretrained(
                "EleutherAI/pythia-70m-deduped",
                revision="step3000",
                device_map=device,
            )
            if not isinstance(model, GPTNeoXForCausalLM):
                raise ValueError("Model is not GPTNeoXForCausalLM")
            self.action_encoder = model.eval()
            for param in self.action_encoder.parameters():
                param.requires_grad = False

            self.encoder = self.encode
        else:
            clip_model, _, _ = open_clip.create_model_and_transforms(vit_config.base_model, vit_config.pretrained)
            self.action_encoder = clip_model.eval()
            for param in self.action_encoder.parameters():
                param.requires_grad = False
            self.encoder = lambda x: self.action_encoder.encode_text(x['input_ids'], normalize=True)

    def encode(self, **kwargs: dict[str, Any]):
        # takes tokens spits out embedding vector
        if not isinstance(self.action_encoder, GPTNeoXForCausalLM):
            raise Exception(
                "yo you trying to call the function that was written for GPTNeoXForCausalLM with something else..."
            )
        out: CausalLMOutputWithPast = self.action_encoder(**kwargs, output_hidden_states=True, return_dict=True)
        last_hidden_state = out.hidden_states[-1]
        return last_token_pool(last_hidden_state, kwargs['attention_mask'])

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(**{"input_ids": input_ids, "attention_mask": attention_mask})


class ContinousActionDecoder(nn.Module):
    def __init__(self, action_set: torch.Tensor):
        """Action set is a set of preembedded actions that have to be searched for the closest action."""
        """ The advantage of encoding actions is that properties and layout of a specific action space is not baked into the model architecture. Only the encoder. """
        super().__init__()
        action_set.requires_grad = False
        self.action_set = action_set.clone().detach()

        self.cosine_sim = nn.CosineSimilarity(dim=-1)

    def forward(self, pred_action: torch.Tensor) -> torch.Tensor:
        """Given the predicted action, find the closest action in the action set."""
        if pred_action.dim() == 2:
            pred_action = pred_action.unsqueeze(1)
        sims = torch.stack(
            [self.cosine_sim(self.action_set.unsqueeze(1), pred_action[:, i, :]) for i in range(pred_action.shape[1])],
            dim=-1,
        )
        # print("sims", sims)
        amax = sims.argmax(dim=0)
        return self.action_set[amax]

    def index(
        self, pred_action: torch.Tensor, return_dist: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Given the predicted action, find the index of the closest action in the action set."""
        logging.debug(f"pred_action shape: {pred_action.shape}")
        if pred_action.dim() == 2:
            pred_action = pred_action.unsqueeze(1)
        sims = torch.stack(
            [self.cosine_sim(self.action_set.unsqueeze(1), pred_action[:, i, :]) for i in range(pred_action.shape[1])],
            dim=-1,
        )  # -> [action_set_size, batch_size, num_actions]
        # print("sims", sims)
        amax = sims.argmax(dim=0)
        if return_dist:
            return amax, sims[amax]
        return amax


# class ContinousActionEncoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.action_encoder = GPTNeoXForCausalLM.from_pretrained(
#             "EleutherAI/pythia-70m-deduped",
#             revision="step3000",
#             # device_map="auto"
#         )
#         for param in self.action_encoder.parameters():
#             param.requires_grad = False

#     def last_token_pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
#         left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
#         if left_padding:
#             return last_hidden_states[:, -1]
#         else:
#             sequence_lengths = attention_mask.sum(dim=1) - 1
#             batch_size = last_hidden_states.shape[0]
#             return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


#     def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, method: EmbeddingMethod = EmbeddingMethod.last_token_pool) -> torch.Tensor:
#         out = self.action_encoder.forward(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
#         last_hidden_state = out.hidden_states[-1]
#         if method == EmbeddingMethod.mean:
#             action = last_hidden_state.mean(dim=1)
#         elif method == EmbeddingMethod.max:
#             action, _ = last_hidden_state.max(dim=1)
#         elif method == EmbeddingMethod.combined:
#             mean_pooled = last_hidden_state.mean(dim=1)
#             max_pooled, _ = last_hidden_state.max(dim=1)
#             action = torch.cat((mean_pooled, max_pooled), dim=1)
#         elif method == EmbeddingMethod.last_token_pool:
#             action = self.last_token_pool(last_hidden_state, attention_mask)
#             # normalize vector
#         action = action / action.norm(dim=1, keepdim=True)
#         return action.float()