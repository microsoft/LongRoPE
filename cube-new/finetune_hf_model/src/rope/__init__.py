import types
import logging
from typing import Optional, Tuple, List, Union

import torch
import numpy as np

from transformers.cache_utils import Cache, DynamicCache
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from .longrope import LlamaLongRoPEScaledRotaryEmbedding, LlamaDynamicLongRoPEScaledRotaryEmbedding, LlamaMixedLongRoPEScaledRotaryEmbedding
from .yarn import LlamaYaRNScaledRotaryEmbedding, LlamaDynamicYaRNScaledRotaryEmbedding
from .ntk import LlamaNTKScaledRotaryEmbedding, LlamaDynamicNTKScalingRotaryEmbedding
import os

logger = logging.getLogger(__name__)


def forward_llama_for_causal_lm(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, CausalLMOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if input_ids is not None and torch.isnan(input_ids).any():
        raise ValueError('input_ids contains nan')
    if attention_mask is not None and torch.isnan(attention_mask).any():
        raise ValueError('attention_mask contains nan')
    if position_ids is not None and torch.isnan(position_ids).any():
        raise ValueError('position_ids contains nan')
    if past_key_values is not None and any(torch.isnan(past_key_value).any() for past_key_value in past_key_values):
        raise ValueError('past_key_values contains nan')
    if inputs_embeds is not None and torch.isnan(inputs_embeds).any():
        raise ValueError('inputs_embeds contains nan')
    if labels is not None and torch.isnan(labels).any():
        raise ValueError('labels contains nan')

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    torch.cuda.empty_cache()

    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states[..., -1:, :]).float()
    if self.mup_width_multiplier:
        logits = logits / self.mup_width_multiplier
    
    if labels is None:
        loss = None
    else:
        loss_fct = torch.nn.CrossEntropyLoss(reduction='sum')
        valid_seq_len = input_ids.shape[-1] - 1
        valid_seq_len_slide_win = torch.sum(labels[:, 1:] >= 0).item()
        loss = 0.0
        for start_idx in range(0, valid_seq_len, 16384):
            end_idx = min(start_idx + 16384, valid_seq_len)
            shift_logits = self.lm_head(hidden_states[..., start_idx:end_idx, :]).float()
            if self.mup_width_multiplier:
                shift_logits = shift_logits / self.mup_width_multiplier
            shift_labels = labels[..., start_idx + 1:end_idx + 1].contiguous()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss += loss_fct(shift_logits, shift_labels)
        loss /= valid_seq_len_slide_win

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def pad_tensor_to_next_mult_of(
    tensor: torch.Tensor,
    dim: int,
    n: int,
) -> Tuple[torch.Tensor, int]:
    """
    Pads a tensor along a specified dimension to the next multiple of a given number.

    Args:
        tensor (torch.Tensor): The input tensor.
        dim (int): The dimension along which to pad the tensor.
        n (int): The number to pad the tensor to the next multiple of.

    Returns:
        Tuple[torch.Tensor, int]: A tuple containing the padded tensor and the amount of padding added.
    """
    residual = tensor.size(dim) % n
    if residual == 0:
        return tensor, 0
    padding = n - residual
    padding_tensor = torch.zeros((*tensor.size()[:dim], padding, *tensor.size()[dim + 1:]), device=tensor.device, dtype=tensor.dtype)
    return torch.cat([tensor, padding_tensor], dim=dim), padding


def strip_padding_from_tensor(
    tensor: torch.Tensor,
    dim: int,
    residual: int,
) -> torch.Tensor:
    """
    Removes padding from a tensor along a specified dimension.

    Args:
        tensor (torch.Tensor): The input tensor.
        dim (int): The dimension along which to remove padding.
        residual (int): The amount of padding to remove.

    Returns:
        torch.Tensor: The tensor with padding removed along the specified dimension.
    """
    return torch.narrow(tensor, dim, 0, tensor.size(dim) - residual)


def forward_llama_model(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[Cache, Tuple[Tuple[torch.FloatTensor]]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    assert not output_attentions
    assert not output_hidden_states
    assert not use_cache

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    past_key_values_length = 0

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0)
    else:
        position_ids = position_ids.view(-1, seq_length).long()
        
    if attention_mask is not None:
        if batch_size <= 0:
            raise ValueError("batch_size has to be defined and > 0")
        logger.warning_once(
            "The `attention_mask` for this model in computed directly inside the attention kernel. A custom attention mask will be ignored ..."
        )
        attention_mask = attention_mask.view(batch_size, -1)
        attention_mask = attention_mask[:, None, None, :]
        attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
    inputs_embeds = self.embedding_dropout(inputs_embeds)

    if self.mup_embedding_multiplier is not None and self.mup_embedding_multiplier > 0.0:
        inputs_embeds = inputs_embeds * self.mup_embedding_multiplier

    residual = 0
    if self.pad_sequence_to_multiple_of_64:
        inputs_embeds, residual = pad_tensor_to_next_mult_of(tensor=inputs_embeds, dim=1, n=64)

    hidden_states = inputs_embeds

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
        hidden_states = layer_outputs[0]

    batch, seq_len, embed_dim = hidden_states.shape
    for start_idx in range(0, seq_len, 16384):
        end_idx = min(seq_len, start_idx + 16384)
        hidden_states[:, start_idx:end_idx, :] = self.final_layernorm(hidden_states[:, start_idx:end_idx, :])

    if residual > 0:
        hidden_states = strip_padding_from_tensor(tensor=hidden_states, dim=1, residual=residual)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)
    
    next_cache = None
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def replace_rope(
    model: AutoModelForCausalLM,
    rope_class: type,
    rope_args: dict,
):
    for idx, layer in enumerate(model.model.layers):
        layer_rope_args = {}
        for k, v in rope_args.items():
            if type(v) is np.ndarray and v.ndim == 2:
                layer_rope_args[k] = v[idx]
            else:
                layer_rope_args[k] = v
        layer_rope_args['dim'] = layer.self_attn.head_dim
        layer_rope_args['device'] = layer.self_attn.rotary_emb.inv_freq.device
        layer.self_attn.rotary_emb = rope_class(**layer_rope_args)
    return model


def load_model(
    model_name_or_path: str,
    rope_method: str,
    max_position_embeddings: int,
    model_class: type = AutoModelForCausalLM,
    config: AutoConfig = None,
    rope_params: dict = None,
    attn_sliding_window: int = None,
    save_memory: int = 0,
    **model_args,
):
    if config is None:
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)

    # NOTE: please force using attn_implementation="flash_attention_2" for now
    if hasattr(config, 'sliding_window') and attn_sliding_window is not None:
        logger.info(f"Change attention sliding window size: {config.sliding_window} => {attn_sliding_window}")
        config.sliding_window = attn_sliding_window

    layer_num = config.num_hidden_layers
    head_size = config.hidden_size // config.num_attention_heads
    half_head_size = head_size // 2

    original_max_position_embeddings = config.max_position_embeddings
    config.max_position_embeddings = max_position_embeddings
    scaling_factor = max_position_embeddings / original_max_position_embeddings

    logger.info(f'[RoPE Method] {rope_method}')
    need_replace_rope = False
    if rope_method == 'pi':
        config.rope_scaling = {'type': 'linear', 'factor': scaling_factor}
    elif rope_method == 'dy_ntk':
        config.rope_scaling = {'type': 'dynamic', 'factor': scaling_factor}
    elif rope_method is not None and rope_method != 'none':
        need_replace_rope = True

    if "torch_dtype" in model_args:
        torch_dtype = model_args["torch_dtype"]
    else:
        torch_dtype = config.torch_dtype
    logger.info(f"[Model Torch Dtype] {torch_dtype}")

    model = model_class.from_pretrained(
        model_name_or_path,
        config=config,
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
        **model_args,
    )

    # check no nan value in model parameters
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            raise ValueError(f'nan value in parameter {name}')

    if save_memory:
        # TODO: move to utils; integrate CUBE for tensor-parallel inference
        model.forward = types.MethodType(forward_llama_for_causal_lm, model)
        model.model.forward = types.MethodType(forward_llama_model, model.model)

    if need_replace_rope:
        rope_class = None
        rope_args = {
            'original_max_position_embeddings': original_max_position_embeddings,
            'max_position_embeddings': max_position_embeddings,
            'scale': scaling_factor,
        }
        if rope_method.startswith('yarn'):
            rope_args['finetuned'] = False
            if rope_method == 'yarn':
                rope_class = LlamaYaRNScaledRotaryEmbedding
            elif rope_method == 'yarn_dynamic':
                rope_class = LlamaDynamicYaRNScaledRotaryEmbedding
        elif rope_method.startswith('longrope'):
            rescale_factors = np.loadtxt(open(rope_params['longrope_params_path'], 'rb'), delimiter=',', skiprows=0)
            if rescale_factors.shape == (half_head_size, ):
                rescale_factors = np.tile(rescale_factors.reshape((1, half_head_size)), (layer_num, 1))
            elif rescale_factors.shape != (layer_num, half_head_size):
                raise ValueError(f'misaligned shape for LongRoPE rescale factors: {rescale_factors.shape}')
            rope_args['rescale_factors'] = rescale_factors
            rope_args['magnitude_scaling_policy'] = rope_params['longrope_scaling_policy']
            # $ cube static LEN for LONGROPE_POSE_SEQ
            if 'LONGROPE_POSE_SEQ' in os.environ:
                rope_args['max_position_embeddings'] = int(os.environ['LONGROPE_POSE_SEQ'])
            
            if rope_method == 'longrope':
                rope_class = LlamaLongRoPEScaledRotaryEmbedding
            elif rope_method == 'longrope_mixed':
                rope_class = LlamaMixedLongRoPEScaledRotaryEmbedding
                rope_args['original_embeddings'] = original_embeddings
                original_rope = model.model.layers[0].self_attn.rotary_emb
                tmp_input = torch.zeros(
                    size=(max_position_embeddings, ),
                    dtype=config.torch_dtype,
                    device=original_rope.inv_freq.device,
                )
                original_embeddings = original_rope(tmp_input)
                rope_args['start_token_idx'] = rope_params['start_token_idx']
            elif rope_method == 'longrope_dynamic':
                rope_class = LlamaDynamicLongRoPEScaledRotaryEmbedding
        if rope_class is None:
            raise ValueError(f'Unsupported RoPE method: {rope_method}')
        logger.info(f'[RoPE Args]{rope_args}')
        return replace_rope(model, rope_class, rope_args)
    else:
        return model
