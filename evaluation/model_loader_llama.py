from typing import Optional, Tuple, List, Union
from argparse import ArgumentParser
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
import transformers
import os
import sys
current_path = os.getcwd()
sys.path.append(current_path)
print(current_path)

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from attention.llama_attn_replace import replace_llama_attn
import math
import numpy as np


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
    assert labels is not None

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
    loss_fct = CrossEntropyLoss(reduction='sum')
    valid_seq_len = input_ids.shape[-1] - 1
    valid_seq_len_slide_win = torch.sum(labels[:, 1:] >= 0).item()
    # print("valid_seq_len_slide_win", valid_seq_len)
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

    return CausalLMOutputWithPast(loss=loss)


def forward_llama_model(
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

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    seq_length_with_past = seq_length
    past_key_values_length = 0

    if past_key_values is not None:
        past_key_values_length = past_key_values[0][0].shape[2]
        seq_length_with_past = seq_length_with_past + past_key_values_length

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
    # embed positions
    if attention_mask is None:
        attention_mask = torch.ones(
            (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
        )
        padding_mask = None
    else:
        if 0 in attention_mask:
            padding_mask = attention_mask
        else:
            padding_mask = None

    attention_mask = self._prepare_decoder_attention_mask(
        attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
    )

    hidden_states = inputs_embeds

    assert not(self.gradient_checkpointing and self.training)

    all_self_attns = None
    all_hidden_states = None

    for idx, decoder_layer in enumerate(self.layers):

        past_key_value = past_key_values[idx] if past_key_values is not None else None

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            padding_mask=padding_mask,
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


def forward_llama_decoder_layer(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    padding_mask: Optional[torch.LongTensor] = None,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    """
    Args:
        hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
        attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
            `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more detail.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
            (see `past_key_values`).
        past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
    """

    residual = hidden_states.clone()
    batch, seq_len, embed_dim = hidden_states.shape

    for start_idx in range(0, seq_len, 16384):
        end_idx = min(seq_len, start_idx + 16384)
        hidden_states[:, start_idx:end_idx, :] = self.input_layernorm(hidden_states[:, start_idx:end_idx, :])
    # print("LN: A({}) R({}) M({})".format(
    #     torch.cuda.memory_allocated(0) / (1024 ** 3),
    #     torch.cuda.memory_reserved(0) / (1024 ** 3),
    #     torch.cuda.max_memory_reserved(0) / (1024 ** 3),
    # ))
    # torch.cuda.empty_cache()

    # Self Attention
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        padding_mask=padding_mask,
    )
    hidden_states = residual + hidden_states
    # print("At: A({}) R({}) M({})".format(
    #     torch.cuda.memory_allocated(0) / (1024 ** 3),
    #     torch.cuda.memory_reserved(0) / (1024 ** 3),
    #     torch.cuda.max_memory_reserved(0) / (1024 ** 3),
    # ))
    # torch.cuda.empty_cache()

    # Fully Connected
    for start_idx in range(0, seq_len, 16384):
        end_idx = min(seq_len, start_idx + 16384)
        part_hidden_states = hidden_states[:, start_idx:end_idx, :].clone()
        part_hidden_states = self.post_attention_layernorm(part_hidden_states)
        part_hidden_states = self.mlp(part_hidden_states)
        hidden_states[:, start_idx:end_idx, :] += part_hidden_states
    # print("FC: A({}) R({}) M({})".format(
    #     torch.cuda.memory_allocated(0) / (1024 ** 3),
    #     torch.cuda.memory_reserved(0) / (1024 ** 3),
    #     torch.cuda.max_memory_reserved(0) / (1024 ** 3),
    # ) + '\n')
    # torch.cuda.empty_cache()

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs


def load_model(model, args):
    # from scaled_rope.modeling_llama import LlamaForCausalLM
    # from scaled_rope.configuration_llama import LlamaConfig

    print("llama config", args.model[0])
    # if "Yarn-Llama-2-7b-64k" in args.model[0][0]:
    #     print("llama config yarn")
    #     from rope.config_llama_yarn import LlamaConfig
    #     config_cls = LlamaConfig
        
    # else:
    from transformers import LlamaConfig
    config_cls = LlamaConfig
    from transformers import LlamaForCausalLM
    model_cls = LlamaForCausalLM
        


    print("aggressive-mem-causal_lm", args.aggressive_mem_causal_lm)
    if args.aggressive_mem_causal_lm:
        transformers.models.llama.modeling_llama.LlamaForCausalLM.forward = forward_llama_for_causal_lm
        transformers.models.llama.modeling_llama.LlamaModel.forward = forward_llama_model

    print("aggressive-mem-decoder", args.aggressive_mem_decoder)
    if args.aggressive_mem_decoder:
        transformers.models.llama.modeling_llama.LlamaDecoderLayer.forward = forward_llama_decoder_layer
        print('Replaced forward functions.')

    print("flash_attn", args.flash_attn)
    if args.flash_attn:
        replace_llama_attn(use_flash_attn=True, use_full=True, inference=True, aggressive_memory=args.aggressive_mem_attn)
        
    model_name = model

    config = config_cls.from_pretrained(
        model_name,
        cache_dir=args.cache_dir,   
    )
    
    scaling_factor = float(args.factor)

    if args.method == "pi":
        print("--use ", args.method, scaling_factor)
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    elif args.method == "dy_ntk":
        print("--use ", args.method)
        config.rope_scaling = {"type": "dynamic", "factor": scaling_factor}

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        cache_dir=args.cache_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        # trust_remote_code=True if "Yarn-Llama-2-7b-64k" in args.model[0][0] else False,
        trust_remote_code=False
    )   
    
    # load rope_para:
    # ft: 4k 8k 256k 512k 1024k 
    if args.finetuned and args.method == "s_pi":
        print("args.finetuned", args.finetuned, "use rope_scale.pt")
        if args.max_tokens != None:
            seq_len = (args.max_tokens + 1023) // 1024
            seq_range = [0, 4, 8, 16, 128, 256, 1024, 2048, 10000]
            for i in range(len(seq_range)-1):
                if seq_range[i] <= seq_len <= seq_range[i+1]:   
                    seq_len = seq_range[i+1]
                    break
            
            if config.model_type == "llama":
                model_type = "la2"
            else:
                raise ValueError("model_type is not llama")  
            ft_model_len = (config.max_position_embeddings + 1023) // 1024

            flag_twice = False
            ft_model_key = None
            
            if seq_len == ft_model_len:
                para_key = f"ft_{model_type}_{ft_model_len}k"
            elif seq_len > ft_model_len:
                para_key = f"{seq_len}k_{model_type}_{ft_model_len}k"
                flag_twice = True
                ft_model_key = f"ft_{model_type}_{ft_model_len}k"
            else:
                para_key = f"{seq_len}k_{model_type}_{ft_model_len}k"
            
            # 128k la2 256k
            if para_key == '128k_la2_256k':
                para_key = 'ft_la2_256k'
                
            rope_rescale = torch.load("./evaluation/rope_rescale.pt")
            # dict_keys(['1024k_la2_128k', '1024k_mis_256k', '2048k_mis_128k', '256k_mis_128k', '512k_mis_128k', '1024k_la2_256k', '2048k_la2_128k', '2048k_mis_256k', '512k_la2_128k', '512k_mis_256k', '1024k_mis_128k', '2048k_la2_256k', '256k_la2_128k', '512k_la2_256k', '16k_la2_128k', '8k_la2_128k', '4k_la2_256k', '8k_mis_128k', '32k_la2_128k', '16k_la2_256k', '8k_la2_256k', '4k_mis_256k', '4k_la2_128k', '32k_la2_256k', '4k_mis_128k', '8k_mis_256k', 'ft_la2_128k', 'ft_la2_256k', 'ft_mis_128k'])
            
            if flag_twice:
                lambda_1 = rope_rescale[para_key] * rope_rescale[ft_model_key]
            else: 
                lambda_1 = rope_rescale[para_key]
        else:
            raise ValueError("args.max_tokens == None")  
    elif args.method == "s_pi" and not args.finetuned:
        print("args.finetuned", args.finetuned, "Not use rope_scale.pt")
        # use base scale
        lambda_1 = np.full((32, 64), 1.0)
    else:
        print("args.finetuned", args.finetuned, "Not use rope_scale.pt")
        lambda_1 = np.full((32, 64), 1.0)
      
    if args.method == "yarn":
        print("--use ", args.method)
        from rope.LlamaYaRNScaledRotaryEmbedding import LlamaYaRNScaledRotaryEmbedding
        print("args.finetuned", args.finetuned)
        for each in model.model.layers:
            each.self_attn.rotary_emb = LlamaYaRNScaledRotaryEmbedding(
                each.self_attn.head_dim, 
                scale=scaling_factor,
                original_max_position_embeddings=args.original_max_position_embeddings, 
                finetuned=args.finetuned, 
                device=each.self_attn.rotary_emb.inv_freq.device
            ) 
    
    elif args.method == "dy_yarn":
        print("--use ", args.method)
        from rope.LlamaDynamicYaRNScaledRotaryEmbedding import LlamaDynamicYaRNScaledRotaryEmbedding
        print("args.finetuned", args.finetuned)
        for each in model.model.layers:
            each.self_attn.rotary_emb = LlamaDynamicYaRNScaledRotaryEmbedding(
                each.self_attn.head_dim, 
                original_max_position_embeddings=args.original_max_position_embeddings, 
                finetuned=args.finetuned, 
                device=each.self_attn.rotary_emb.inv_freq.device
            ) 
    elif args.method == "s_pi":
        print("--use ", args.method)
        from rope.LlamaSPIScaledRotaryEmbedding import LlamaSPIScaledRotaryEmbedding
        print("args.finetuned", args.finetuned)
        
        # lambda_1 = np.loadtxt(open(args.s_pi_para, "rb"), delimiter=",", skiprows=0)
        # if args.s_pi_twice_para != None:
        #     lambda_twice = np.loadtxt(open(args.s_pi_twice_para, "rb"), delimiter=",", skiprows=0)
        #     if lambda_twice.shape == (64,):
        #         lambda_twice = np.tile(lambda_twice, (32, 1))
        #     assert lambda_twice.shape == (32, 64), f"lambda_twice shape error {lambda_twice.shape}"
        #     lambda_1 = lambda_1 * lambda_twice
            
        assert lambda_1.shape == (32, 64), f"lambda_1 shape error {lambda_1.shape}"
        
        layer = 0
        for each in model.model.layers:
            each.self_attn.rotary_emb = LlamaSPIScaledRotaryEmbedding(
                each.self_attn.head_dim, 
                scale=scaling_factor,
                original_max_position_embeddings=args.original_max_position_embeddings, 
                finetuned=args.finetuned, 
                device=each.self_attn.rotary_emb.inv_freq.device,
                lambda_1=lambda_1[layer, :],
                tmps=args.tmps
            ) 
            layer += 1
    
    elif args.method == "s_pi_start":
        print("--use ", args.method)
        from rope.LlamaSPIScaledStartTokenRotaryEmbedding import LlamaSPIScaledStartTokenRotaryEmbedding
        print("args.finetuned", args.finetuned)
        
        # lambda_1 = np.loadtxt(open(args.s_pi_para, "rb"), delimiter=",", skiprows=0)
        
        assert lambda_1.shape == (32, 64), f"lambda_1 shape error {lambda_1.shape}"
        
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
        
        seq_len = args.max_tokens
        tmp_device = "cpu"
        rotary_emb_origin = LlamaRotaryEmbedding(dim=model.model.layers[0].self_attn.head_dim, max_position_embeddings=seq_len, device=tmp_device)
        input_x = torch.zeros((1,),dtype=torch.float16, device=tmp_device)
        cos_sin_origin = rotary_emb_origin.forward(x=input_x, seq_len=seq_len)
        # cos_sin_origin=None
        
        layer = 0
        print("start_token", args.stream)
        for each in model.model.layers:
            each.self_attn.rotary_emb = LlamaSPIScaledStartTokenRotaryEmbedding(
                each.self_attn.head_dim, 
                scale=scaling_factor,
                original_max_position_embeddings=args.original_max_position_embeddings, 
                finetuned=args.finetuned, 
                device=each.self_attn.rotary_emb.inv_freq.device,
                lambda_1=lambda_1[layer, :],
                tmps=args.tmps,
                start_token=args.stream,
                cos_sin_origin=cos_sin_origin
            ) 
            layer += 1
        
    elif args.method == "dy_s_pi":
        print("--use ", args.method)
        from rope.LlamaDynamicSPIScaledRotaryEmbedding import LlamaDynamicSPIScaledRotaryEmbedding
        print("args.finetuned", args.finetuned)
        
        # lambda_1 = np.loadtxt(open(args.s_pi_para, "rb"), delimiter=",", skiprows=0)
        
        assert lambda_1.shape == (32, 64), f"lambda_1 shape error {lambda_1.shape}"
        layer = 0
        for each in model.model.layers:
            each.self_attn.rotary_emb = LlamaDynamicSPIScaledRotaryEmbedding(
                each.self_attn.head_dim, 
                scale=scaling_factor,
                original_max_position_embeddings=args.original_max_position_embeddings, 
                finetuned=args.finetuned, 
                device=each.self_attn.rotary_emb.inv_freq.device,
                lambda_1=lambda_1[layer, :],
                tmps=args.tmps
            ) 
            layer += 1
            
    elif args.method == "ntk":   
        print("--use ", args.method)
        print("Use ntk for code llama")
        print("config.rope_theta", config.rope_theta)
        from rope.LlamaCodeLlamaScaledRotaryEmbedding import LlamaCodeLlamaScaledRotaryEmbedding

        for each in model.model.layers:
            each.self_attn.rotary_emb = LlamaCodeLlamaScaledRotaryEmbedding(
                each.self_attn.head_dim, 
                device=each.self_attn.rotary_emb.inv_freq.device,
                scaling_factor=scaling_factor,
                rope_theta=config.rope_theta
            ) 
    elif args.method not in ["pi", "dy_ntk"]:
        raise ValueError(
                f"No support {args.method}"
            )
                  
    return model, lambda_1


def add_args(parser: ArgumentParser):
    
    parser.add_argument("--max-position-embeddings", type=int)
    parser.add_argument("--original-max-position-embeddings", type=int)
    # parser.add_argument("--flash-attention", action="store_true")
    parser.add_argument("--cache_dir", type=str)
    parser.add_argument("--flash_attn", action="store_true")
    parser.add_argument("--method", type=str, default="pi")
    parser.add_argument("--s_pi_para", type=str, default="./evolution/dim_mono/result_alpha/dim_mono_8192_result.csv")
    parser.add_argument("--s_pi_twice_para", type=str, default=None)
    parser.add_argument("--tmps", type=str, default="su", help='')
    parser.add_argument("--factor", type=float)
    parser.add_argument("--finetuned", action="store_true")
    parser.add_argument("--aggressive-mem-decoder", action="store_true")
    parser.add_argument("--aggressive-mem-causal_lm", action="store_true")
    parser.add_argument("--aggressive-mem-attn", action="store_true")
    parser.add_argument("--stream", type=int, default=0)
    parser.add_argument("--peft-model", type=str)
    parser.add_argument("--use_cache", action="store_true")
    return parser
# 


def load_model_and_apply_patches(model, args):
    # print(args)
    # return apply_patches(load_model(model, args), args)
    return load_model(model, args)
