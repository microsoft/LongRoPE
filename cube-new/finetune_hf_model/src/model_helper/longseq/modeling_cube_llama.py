from typing import Optional, Tuple
import warnings

import torch
from torch import Tensor
import torch.nn.functional as F
import os

from transformers.models.llama.modeling_llama import (
    LlamaConfig,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaForCausalLM,
    LlamaAttention,
    LlamaMLP,
    LlamaRMSNorm,
    LlamaDecoderLayer,
    Cache,
    logger,
    apply_rotary_pos_emb,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    _get_unpad_data,
    LLAMA_ATTENTION_CLASSES
)

from cube.graph.parser import register
from apex.normalization.fused_layer_norm import fused_rms_norm_affine

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

# 
if 'ATTENTION_TYPE' in os.environ and os.environ['ATTENTION_TYPE'] == "ring_attn":
    try:
        from zigzag_ring_attention.zigzag_attn import wrap_zigzag_attn_func
        
        ''' Use `^` to avoid split this para '''
        # only spilt seq len
        def zigzag_attn_func(q: Tensor, k: Tensor, v: Tensor, softmax_scale: Tensor=None,
                             dropout_p: float=0.0, causal: bool=True, window_size: Tuple[int]=(-1, -1),
                             alibi_slopes: Tensor=None, deterministic: bool=False,
                             return_attn_probs: bool=False,
                             process_group: Tuple[int]=None) -> Tensor:
            return wrap_zigzag_attn_func(
                q, k, v, softmax_scale, dropout_p, causal, window_size, alibi_slopes, deterministic, return_attn_probs, process_group
            )
        register('bs^ ql h^ dim^, bs^ ql h^ dim^, bs^ ql h^ dim^ -> bs^ ql h^ dim^')(zigzag_attn_func)
        print("$ use ring attn")
    except:
        raise ImportError("Can not import `wrap_zigzag_attn_func` from `examples.zigzag_ring_attention.zigzag_attn`")
else:
    print("$ use flash attn")
class CubeLlamaFlashAttention2(LlamaAttention):
    """
    Llama flash attention module. This module inherits from `LlamaAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # LlamaFlashAttention2 attention does not support output_attentions
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

            # overwrite attention_mask with padding_mask
            attention_mask = kwargs.pop("padding_mask")

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        
        print("$cos", cos.shape, "\nposition_ids", position_ids.shape, position_ids[0,:5], position_ids[0,-5:])
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            # Handle the case where the model is quantized
            if hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
            causal = self.is_causal and q_len != 1

        # print("$1 query_states.device", query_states.device)
        if 'ATTENTION_TYPE' in os.environ and os.environ['ATTENTION_TYPE'] == "ring_attn":
            # # Use ring attention --> wrap_zigzag_attn_func
            attn_output = zigzag_attn_func(
                query_states, key_states, value_states, dropout_p=dropout_rate, causal=causal
            )
            print("$ring attn")
        else:
            attn_output = cube_llama_flash_attention_forward(
                query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate, num_heads=self.num_heads, causal=causal
            )
            print("$flash attn")

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

def cube_llama_flash_attention_forward(
    query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None, num_heads=None, causal=True
):
    """
    Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
    first unpad the input, then computes the attention scores and pad the final attention scores.

    Args:
        query_states (`torch.Tensor`):
            Input query states to be passed to Flash Attention API
        key_states (`torch.Tensor`):
            Input key states to be passed to Flash Attention API
        value_states (`torch.Tensor`):
            Input value states to be passed to Flash Attention API
        attention_mask (`torch.Tensor`):
            The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
            position of padding tokens and 1 for the position of non-padding tokens.
        dropout (`int`, *optional*):
            Attention dropout
        softmax_scale (`float`, *optional*):
            The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
    """
    # Contains at least one padding token in the sequence
    if attention_mask is not None:
        batch_size = query_states.shape[0]
        query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = _upad_input(
            query_states, key_states, value_states, attention_mask, query_length, num_heads
        )

        cu_seqlens_q, cu_seqlens_k = cu_seq_lens
        max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

        attn_output_unpad = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_in_batch_q,
            max_seqlen_k=max_seqlen_in_batch_k,
            dropout_p=dropout,
            softmax_scale=softmax_scale,
            causal=causal,
        )

        attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
    else:
        attn_output = flash_attn_func(
            query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
        )

    return attn_output

def _upad_input(query_layer, key_layer, value_layer, attention_mask, query_length, num_heads):
    indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
    batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

    key_layer = index_first_axis(
        key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
    )
    value_layer = index_first_axis(
        value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
    )
    if query_length == kv_seq_len:
        query_layer = index_first_axis(
            query_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k
        )
        cu_seqlens_q = cu_seqlens_k
        max_seqlen_in_batch_q = max_seqlen_in_batch_k
        indices_q = indices_k
    elif query_length == 1:
        max_seqlen_in_batch_q = 1
        cu_seqlens_q = torch.arange(
            batch_size + 1, dtype=torch.int32, device=query_layer.device
        )  # There is a memcpy here, that is very bad.
        indices_q = cu_seqlens_q[:-1]
        query_layer = query_layer.squeeze(1)
    else:
        # The -q_len: slice assumes left padding.
        attention_mask = attention_mask[:, -query_length:]
        query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

    return (
        query_layer,
        key_layer,
        value_layer,
        indices_q,
        (cu_seqlens_q, cu_seqlens_k),
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
    )

if not ('ATTENTION_TYPE' in os.environ and os.environ['ATTENTION_TYPE'] == "ring_attn"):
    # SFT setting: Use mask
    # register('b l^ num_heads hd^, b s^ num_heads hd^, b s^ num_heads vd^, b l^ -> b l^ num_heads vd^')(cube_llama_flash_attention_forward)

    # FT setting: None mask
    register('b l^ num_heads hd^, b s^ num_heads hd^, b s^ num_heads vd^ -> b l^ num_heads vd^')(cube_llama_flash_attention_forward)

# FT setting: Use ring attn: register on single file


LLAMA_ATTENTION_CLASSES['flash_attention_2'] = CubeLlamaFlashAttention2


def cube_recompute_llama_ffn_forward(x: torch.Tensor, fc1: torch.Tensor, fc2: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return F.linear(F.silu(F.linear(x, fc1, None)) * F.linear(x, gate, None), fc2, None)

register('* h^, i+ h^, h^ i+, i+ h^ -> * h^')(cube_recompute_llama_ffn_forward)


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
            # down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
            down_proj = cube_recompute_llama_ffn_forward(x, self.gate_proj.weight, self.down_proj.weight, self.up_proj.weight)

        return down_proj


class CubeLlamaRMSNorm(LlamaRMSNorm):
    def forward(self, hidden_states):
        return fused_rms_norm_affine(hidden_states, self.weight, (hidden_states.shape[-1],), self.variance_epsilon)


class CubeLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        torch.nn.Module.__init__(self)
        self.hidden_size = config.hidden_size

        self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = CubeLlamaMLP(config)
        self.input_layernorm = CubeLlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = CubeLlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class CubeLlamaModel(LlamaModel):
    def __init__(self, config: LlamaConfig):
        LlamaPreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = torch.nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = torch.nn.ModuleList(
            [CubeLlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._use_sdpa = config._attn_implementation == "sdpa"
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self.norm = CubeLlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
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
