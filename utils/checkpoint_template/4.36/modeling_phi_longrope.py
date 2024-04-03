import math
from typing import Optional

import torch
from torch import nn

from transformers.utils import logging
# LLAMA COMPONENTS BEGIN
from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm as BaseRMSNorm,
    LlamaRotaryEmbedding as BaseRotaryEmbedding,
    LlamaMLP as BaseMLP,
    LlamaAttention as BaseAttention,
    LlamaFlashAttention2 as BaseFlashAttention2,
    LlamaSdpaAttention as BaseSdpaAttention,
    LlamaDecoderLayer as BaseDecoderLayer,
    LlamaModel as BaseModel,
    LlamaForCausalLM as BaseModelForCausalLM,
    LlamaForSequenceClassification as BaseModelForSequenceClassification,
)
# LLAMA COMPONENTS END
# MISTRAL COMPONENTS BEGIN
from transformers.models.mistral.modeling_mistral import (
    MistralRMSNorm as BaseRMSNorm,
    MistralRotaryEmbedding as BaseRotaryEmbedding,
    MistralMLP as BaseMLP,
    MistralAttention as BaseAttention,
    MistralFlashAttention2 as BaseFlashAttention2,
    MistralFlashAttention2 as BaseSdpaAttention,
    MistralDecoderLayer as BaseDecoderLayer,
    MistralModel as BaseModel,
    MistralForCausalLM as BaseModelForCausalLM,
    MistralForSequenceClassification as BaseModelForSequenceClassification,
)
# MISTRAL COMPONENTS END

from .configuration_phi_longrope import PhiLongRoPEConfig


logger = logging.get_logger(__name__)


class PhiLongRoPEScaledRotaryEmbedding(BaseRotaryEmbedding):

    def __init__(
        self,
        dim, 
        short_factor,
        long_factor,
        max_position_embeddings=4096,
        original_max_position_embeddings=4096,
        base=10000,
        magnitude_scaling_policy="su",
        device=None,
    ):
        super(BaseRotaryEmbedding, self).__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.original_max_position_embeddings = original_max_position_embeddings
        self.base = base

        if magnitude_scaling_policy == "su":
            self._calc_mscale = self._calc_mscale_su
        elif magnitude_scaling_policy == "yarn":
            self._calc_mscale = self._calc_mscale_yarn
        else:
            self._calc_mscale = lambda scale: float(scale)

        self.short_factor = short_factor
        self.long_factor = long_factor

    def _calc_mscale_su(self, scale):
        if scale <= 1.0:
            return 1.0
        return math.sqrt(1 + math.log(scale) / math.log(self.original_max_position_embeddings))

    def _calc_mscale_yarn(self, scale):
        if scale <= 1.0:
            return 1.0
        return 0.1 * math.log(scale) + 1.0

    @torch.no_grad()
    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[-2]

        if seq_len > self.original_max_position_embeddings:
            t = torch.arange(seq_len, device=x.device, dtype=torch.float32)
            rescale_factors = torch.tensor(self.long_factor, dtype=torch.float32, device=x.device)
        else:
            t = torch.arange(self.original_max_position_embeddings, device=x.device, dtype=torch.float32)
            rescale_factors = torch.tensor(self.short_factor, dtype=torch.float32, device=x.device)
        assert rescale_factors.shape == (self.dim // 2, ), \
            f"misaligned shape for LongRoPE rescale factors: {rescale_factors.shape}"

        inv_freq = 1.0 / (rescale_factors * (self.base ** (torch.arange(0, self.dim, 2).float().to(x.device) / self.dim)))

        freqs = torch.outer(t, inv_freq)

        mscale = self._calc_mscale(self.max_position_embeddings / self.original_max_position_embeddings)

        emb = torch.cat((freqs, freqs), dim=-1)
        return (emb.cos() * mscale).to(x.dtype), (emb.sin() * mscale).to(x.dtype)


class PhiLongRoPEAttention(BaseAttention):

    # MISTRAL COMPONENTS BEGIN
    def __init__(self, config: PhiLongRoPEConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self._init_rope()
    # MISTRAL COMPONENTS END

    def _init_rope(self):
        if self.config.rope_scaling is None:
            logger.warning_once(
                "Instantiating RoPE using original Llama / Mistral Rotary Embedding."
            )
            self.rotary_emb = BaseRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            short_factor = self.config.rope_scaling["short_factor"]
            long_factor = self.config.rope_scaling["long_factor"]
            self.rotary_emb = PhiLongRoPEScaledRotaryEmbedding(
                dim=self.head_dim, 
                short_factor=short_factor,
                long_factor=long_factor,
                max_position_embeddings=self.config.max_position_embeddings,
                original_max_position_embeddings=self.config.original_max_position_embeddings,
                base=self.config.rope_theta,
                device=self.q_proj.weight.device,
            )

class PhiLongRoPEFlashAttention2(PhiLongRoPEAttention, BaseFlashAttention2):

    pass


class PhiLongRoPESdpaAttention(PhiLongRoPEAttention, BaseSdpaAttention):

    pass


PHI_LONGROPE_ATTENTION_CLASSES = {
    "eager": PhiLongRoPEAttention,
    "flash_attention_2": PhiLongRoPEFlashAttention2,
    "sdpa": PhiLongRoPESdpaAttention,
}


class PhiLongRoPEDecoderLayer(BaseDecoderLayer):
    def __init__(self, config: PhiLongRoPEConfig, layer_idx: int):
        super(BaseDecoderLayer, self).__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = PHI_LONGROPE_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = BaseMLP(config)
        self.input_layernorm = BaseRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = BaseRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class PhiLongRoPEModel(BaseModel):

    def __init__(self, config: PhiLongRoPEConfig):
        super(BaseModel, self).__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [PhiLongRoPEDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._use_sdpa = config._attn_implementation == "sdpa"
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self.norm = BaseRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()


class PhiLongRoPEForCausalLM(BaseModelForCausalLM):

    def __init__(self, config):
        super(BaseModelForCausalLM, self).__init__(config)
        self.model = PhiLongRoPEModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()


class PhiLongRoPEForSequenceClassification(BaseModelForSequenceClassification):

    def __init__(self, config):
        super(BaseModelForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.model = BaseModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
