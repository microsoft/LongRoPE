from typing import List, Optional, Tuple, Union
import cube
import torch
import torch.utils.checkpoint
from torch import nn

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from flash_attn import flash_attn_func, flash_attn_with_kvcache

# fused kernels
from xformers.ops.rmsnorm import rms_norm
from xformers.ops.swiglu_op import swiglu
from flash_attn.layers.rotary import apply_rotary_emb_func

@cube.graph.parser.register('N L E^, E^ -> N L E^')
def RmsNorm(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    LlamaRMSNorm
    """
    torch.cuda.empty_cache()
    if hidden_states.shape[1] == 1:
        dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + eps)
        output = (weight * hidden_states).to(dtype)
        return output
    else:
        # return rms_norm(hidden_states, weight, eps)
        if hidden_states.shape[1] > 512 * 1024:
            shard_size = 16
        else:
            shard_size = 4
        output = torch.empty_like(hidden_states)
        shard_len = hidden_states.shape[1] // shard_size + 1
        for start_loc in range(0, hidden_states.shape[1], shard_len):
            end_loc = min(start_loc + shard_len, hidden_states.shape[1])
            shard = hidden_states[:, start_loc:end_loc, :].clone()
            shard = rms_norm(shard, weight, eps)
            output[:, start_loc:end_loc, :].copy_(shard)
        return output


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return RmsNorm(hidden_states, self.weight, self.variance_epsilon)

class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

@cube.graph.parser.register('bs seq_len hidden_size^, inter_size+ hidden_size^, \
    inter_size+ hidden_size^, hidden_size^ inter_size+ -> bs seq_len^ hidden_size^')
def ffn(x: torch.Tensor,
        gate: torch.Tensor, 
        up: torch.Tensor,
        down: torch.Tensor) -> torch.Tensor:
    # return torch.empty_like(x)
    if x.shape[1] == 1:
        y = torch.nn.functional.linear(x, up)
        x = torch.nn.functional.silu(torch.nn.functional.linear(x, gate))
        x = x * y
        return torch.nn.functional.linear(x, down)
    else:
        # return swiglu(x=x, w1=gate, b1=None, w2=up, b2=None, w3=down, b3=None, op=None)
        # shard
        if x.shape[1] > 512 * 1024:
            shard_size = 16
        else:
            shard_size = 4
        shard_len = x.shape[1] // shard_size + 1
        for start_loc in range(0, x.shape[1], shard_len):
            end_loc = min(start_loc + shard_len, x.shape[1])
            shard = x[:, start_loc:end_loc, :].clone()
            shard = swiglu(x=shard, w1=gate, b1=None, w2=up, b2=None, w3=down, b3=None, op=None)
            x[:, start_loc:end_loc, :].copy_(shard)
        return x

class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        return ffn(x, self.gate_proj.weight, self.up_proj.weight, self.down_proj.weight)

@cube.graph.parser.register('bs q_len head_num head_dim^, \
    bs q_len head_num head_dim^, bs q_len, a^ length^ b^ dim^, \
        a^ length^ b^ dim^ -> bs q_len head_num head_dim^, bs q_len head_num head_dim^')
def apply_rotary_pos_emb_inference(q, k, position_ids, cos_cache, sin_cache):
    '''
    # no kv_cache version
    gather_indices = position_ids[:, :, None, None]  # [bsz, seq_len, 1, 1]
    gather_indices = gather_indices.repeat(
        1, 1, cos_cache.shape[1], cos_cache.shape[3] # [bsz, seq_len, 1, dim]
    )
    bsz = gather_indices.shape[0]
    cos = torch.gather(cos_cache.transpose(1, 2).repeat(bsz, 1, 1, 1), 1, gather_indices) 
    sin = torch.gather(sin_cache.transpose(1, 2).repeat(bsz, 1, 1, 1), 1, gather_indices)
    q = q * cos + rotate_half(q) * sin # [bs seq_len (kv_head_num 4) head_dim] [bs seq_len 1 head_dim]
    k = k * cos + rotate_half(k) * sin # [bs seq_len kv_head_num head_dim] [bs seq_len 1 head_dim]
    '''
    
    # apply_rotary_emb_func       
    bsz, seq_len, _, head_dim = q.shape
    if seq_len > 1:
        half_cos = cos_cache.squeeze()
        half_sin = sin_cache.squeeze()
        q = apply_rotary_emb_func(q, half_cos, half_sin, interleaved=False)
        k = apply_rotary_emb_func(k, half_cos, half_sin, interleaved=False)
    else:
        cos = cos_cache.repeat(bsz, 1, 1, 1) # [bs seq_len 1 head_dim]
        sin = sin_cache.repeat(bsz, 1, 1, 1) # [bs seq_len 1 head_dim]
        q = q * cos + rotate_half(q) * sin # [bs seq_len (kv_head_num 4) head_dim] 
        k = k * cos + rotate_half(k) * sin # [bs seq_len kv_head_num head_dim] 
    return q, k

@cube.graph.parser.register('bs q_length num_heads head_dim, \
    bs seq_length num_heads head_dim, \
        bs seq_length num_heads head_dim \
            -> bs q_length (num_heads head_dim)')
def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bsz: int,
    q_len: int,
    sliding_window_attention: int,
):
    # return torch.empty_like(q).view(bsz, q_len, -1)
    return flash_attn_func(q, k, v, 0.0, None, True, (sliding_window_attention, sliding_window_attention)).view(
        bsz, q_len, -1
    )  

@cube.graph.parser.register('bs q_length num_heads head_dim, \
    bs seq_length tp_num_heads^ head_dim, \
        bs seq_length tp_num_heads^ head_dim, \
            bs q_length num_heads head_dim, \
                bs q_length num_heads head_dim \
            -> bs q_length (num_heads head_dim)')
def flash_attention_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    target_length: int,
    bsz: int,
    q_len: int,
    sliding_window_attention: int,
):
    # return torch.empty_like(q).view(bsz, q_len, -1)
    if q_len > 1:
        k_cache[:, :q_len, :, :].copy_(k)
        v_cache[:, :q_len, :, :].copy_(v)
        return flash_attn_func(q, k, v, 0.0, None, True, (sliding_window_attention, sliding_window_attention)).view(
            bsz, q_len, -1
        ) 
    else:
        return flash_attn_with_kvcache(q=q, k_cache=k_cache, v_cache=v_cache, k=k, v=v, cache_seqlens=target_length, softmax_scale=None, causal=True, window_size=(sliding_window_attention, sliding_window_attention)).view(
            bsz, q_len, -1
        )  

class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, 
                 config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        method: str = "pi",
        tmps: str = "non",
        lambda_1: torch.Tensor = None,
        scaling_factor: float = 1.0,
        finetuned: bool = False,
        start_token: int = 0,
        sliding_window_attention: int = 2048,
        target_length: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()
        kv_heads = self.num_key_value_heads

        if past_key_value is not None:
            kv_seq_len = q_len + target_length
        else:
            kv_seq_len = q_len
            
        cos_cache, sin_cache = self.rotary_emb(
            seq_len=kv_seq_len,
            method=method,
            tmps=tmps,
            lambda_1=lambda_1,
            scaling_factor=scaling_factor,
            finetuned=finetuned,
            start_token=start_token,
            position_ids=position_ids,
            )
        
        q, k, v = (
            op(hidden_states).view(bsz, q_len, nh, self.head_dim)
            for op, nh in (
                (self.q_proj, self.num_heads),
                (self.k_proj, kv_heads),
                (self.v_proj, kv_heads),
            )
        )
        
        q, k = apply_rotary_pos_emb_inference(q, k, position_ids, cos_cache, sin_cache)

        if past_key_value is not None:
            output = flash_attention_with_kvcache(q, past_key_value[0], past_key_value[1], k, v, target_length, bsz, q_len, sliding_window_attention)
        else:
            output = flash_attention(q, k, v, bsz, q_len, sliding_window_attention)        
        
        return self.o_proj(output)
        
        

class LlamaDecoderLayer(nn.Module):
    def __init__(self, 
                 config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = (LlamaAttention(config=config))
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        method: str = "pi",
        tmps: str = "non",
        lambda_1: torch.Tensor = None,
        scaling_factor: float = 1.0,
        finetuned: bool = False,
        start_token: int = 0,
        sliding_window_attention: int = 2048,
        target_length: int = 0,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        residual = hidden_states
        
        hidden_states = self.input_layernorm(hidden_states)
        
        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_ids=position_ids,
            past_key_value=past_key_value,
            method=method,
            tmps=tmps,
            lambda_1=lambda_1,
            scaling_factor=scaling_factor,
            finetuned=finetuned,
            start_token=start_token,
            sliding_window_attention=sliding_window_attention,
            target_length=target_length,
        )
        
        # hidden_states = residual + hidden_states
        hidden_states.add_(residual)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        # hidden_states = residual + hidden_states
        hidden_states.add_(residual)

        outputs = (hidden_states,)

        return outputs
    
class LlamaModel(nn.Module):

    def __init__(self, 
                 config: LlamaConfig):
        super().__init__()
        self.connfig = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
    
    def get_data(
        self,
        idx: int,
        method: List[str],
        tmps: List[str],
        lambda_1: torch.Tensor,
        scaling_factor: List[float],
        finetuned: List[bool],
        start_token: List[int],
        sliding_window_attention: List[int],
        ):
        return method[idx], tmps[idx], lambda_1[idx], scaling_factor[idx], finetuned[idx], start_token[idx], sliding_window_attention[idx]

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        method: List[str] = None,
        tmps: List[str] = None,
        lambda_1: torch.Tensor = None,
        scaling_factor: List[float] = None,
        finetuned: List[bool] = None,
        start_token: List[int] = None,
        sliding_window_attention: List[int] = None,
        target_length: int = 0,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
            
        _, seq_length = input_ids.shape

        if past_key_values is not None:
            past_key_values_length = target_length
        else:
            past_key_values_length = 0
            
        # position_ids = prepare_position_ids(input_ids, seq_length, past_key_values_length)
        device = input_ids.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        
        hidden_states = self.embed_tokens(input_ids)

        # decoder layers
        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            i_method, i_tmp, i_lambda_1, i_scaling_factor, i_finetuned, i_start_token, i_sliding_window_attention = self.get_data(idx, method, tmps, lambda_1, scaling_factor, finetuned, start_token, sliding_window_attention)
            layer_outputs = decoder_layer(
                hidden_states,
                position_ids=position_ids,
                past_key_value=past_key_value,
                method=i_method,
                tmps=i_tmp,
                lambda_1=i_lambda_1,
                scaling_factor=i_scaling_factor,
                finetuned=i_finetuned,
                start_token=i_start_token,
                sliding_window_attention=i_sliding_window_attention,
                target_length=target_length,
            )

            hidden_states = layer_outputs[0]
        
        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
        )

@cube.graph.parser.register('bs seq_len hidden_size -> bs hidden_size')
def cut_len(hidden_states: torch.Tensor, length: int):
    return hidden_states[:, length-1:length, :].view(hidden_states.shape[0], -1)

class LlamaForCausalLM(nn.Module):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        print("Use Local Llama Implementation")

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        method: List[str] = None,
        tmps: List[str] = None,
        lambda_1: torch.Tensor = None,
        scaling_factor: List[float] = None,
        finetuned: List[bool] = None,
        start_token: List[int] = None,
        cache_length: List[int] = None,
        sliding_window_attention: List[int] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        post_length = cache_length[0]
        target_length = input_ids.shape[1]
        
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            method=method,
            tmps=tmps,
            lambda_1=lambda_1,
            scaling_factor=scaling_factor,
            finetuned=finetuned,
            start_token=start_token,
            sliding_window_attention=sliding_window_attention,
            target_length=post_length,
        )

        hidden_states = outputs[0]

        # logits = self.lm_head(hidden_states[..., len-1:len, :]).float()
        hidden_states = cut_len(hidden_states, target_length)
        logits = self.lm_head(hidden_states.unsqueeze(1)).float()
        return CausalLMOutputWithPast(
            logits=logits,
        )   

    def load_weight(self,
                    model_name_or_path: str,
                    cache_dir: Optional[str] = None,
                    use_np_cache: bool = False):
        state_dict = self.state_dict()
        name_list = ['inv_freq', 'cos_cached', 'sin_cached']
        from evaluation.model_executor.weight_utils import hf_model_weights_iterator
        
        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, use_np_cache):
            skip = False
            for n in name_list:
                if n in name:
                    skip = True
            if skip:
                continue
            param = state_dict[name]
            param.data.copy_(loaded_weight)
        