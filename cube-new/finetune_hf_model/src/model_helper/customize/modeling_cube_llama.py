import torch
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import (
    LlamaConfig,
    LlamaRMSNorm,
    LlamaMLP,
    LlamaDecoderLayer,
    LlamaModel,
    LlamaForCausalLM,
    LlamaPreTrainedModel,
)
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

from cube.graph.parser import register

from .modeling_cube_llama_attention import CUBE_LLAMA_ATTENTION_CLASSES


def cube_rms_norm(hidden_states: torch.Tensor, weight: torch.Tensor, eps: int = 1e-6):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return weight * hidden_states.to(input_dtype)

register('* h^, h^ -> * h^', 'cube_rms_norm')(cube_rms_norm)


def cube_ffn_forward(x: torch.Tensor, fc1: torch.Tensor, fc2: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return F.linear(F.silu(F.linear(x, fc1, None)) * F.linear(x, gate, None), fc2, None)

register('* h^, i+ h^, h^ i+, i+ h^ -> * h^', 'cube_ffn_forward')(cube_ffn_forward)


class CubeLlamaRMSNorm(LlamaRMSNorm):
    def forward(self, hidden_states):
        return cube_rms_norm(hidden_states, self.weight, self.variance_epsilon)


ALL_LAYERNORM_LAYERS.append(CubeLlamaRMSNorm)


class CubeLlamaMLP(LlamaMLP):
    def __init__(self, config):
        if config.hidden_act != "silu":
            raise ValueError(f"Only silu activation is supported for now, got {config.hidden_act}")
        super().__init__(config)

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = cube_ffn_forward(x, self.gate_proj.weight, self.down_proj.weight, self.up_proj.weight)

        return down_proj


class CubeLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = CUBE_LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = CubeLlamaMLP(config)
        self.input_layernorm = CubeLlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = CubeLlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class CubeLlamaModel(LlamaModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        LlamaPreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = torch.nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = torch.nn.ModuleList(
            [CubeLlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = CubeLlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        # register a causal mask to separate causal and padding mask creation. Merging happends in the attention class
        causal_mask = torch.full((config.max_position_embeddings, config.max_position_embeddings), fill_value=1)
        self.register_buffer("causal_mask", torch.triu(causal_mask, diagonal=1), persistent=False)
        # Initialize weights and apply final processing
        self.post_init()



class CubeLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        LlamaPreTrainedModel.__init__(self, config)
        self.model = CubeLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
