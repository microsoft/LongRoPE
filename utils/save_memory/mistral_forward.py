# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import logging
from typing import Optional, Tuple, List, Union

import torch
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast


logger = logging.getLogger(__name__)


def forward_mistral_for_causal_lm(
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
            shift_labels = labels[..., start_idx + 1:end_idx + 1].contiguous()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss += loss_fct(shift_logits, shift_labels)
        loss /= valid_seq_len_slide_win

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def forward_mistral_model(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    assert not output_attentions
    assert not output_hidden_states
    assert not use_cache
    assert not self.training

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

    past_key_values_length = 0

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    else:
        position_ids = position_ids.view(-1, seq_length).long()

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if self._attn_implementation == "flash_attention_2":
        # 2d mask is passed through the layers
        attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
    else:
        raise ValueError("attn_implementation should be `flash_attention_2`")

    hidden_states = inputs_embeds

    # decoder layers
    all_hidden_states = None
    all_self_attns = None
    next_decoder_cache = None

    for decoder_layer in self.layers:
        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = layer_outputs[0]

    batch, seq_len, embed_dim = hidden_states.shape
    for start_idx in range(0, seq_len, 16384):
        end_idx = min(seq_len, start_idx + 16384)
        hidden_states[:, start_idx:end_idx, :] = self.norm(hidden_states[:, start_idx:end_idx, :])

    next_cache = None
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )
