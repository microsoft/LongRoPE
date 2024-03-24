from typing import Optional, Tuple, List, Union
from argparse import ArgumentParser
from transformers.modeling_outputs import CausalLMOutputWithPast
import transformers
import os
import sys
current_path = os.getcwd()
sys.path.append(current_path)
print(current_path)

# from scaled_rope.patch import *
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from attention.mistral_attn_replace import replace_mistral_attn, forward_mistral_for_causal_lm, forward_mistral_decoder_layer
import math
import numpy as np


def load_model(model, args):
    from transformers import MistralForCausalLM
    if "Yarn-Mistral-7b-128k" in args.model[0][0] or "Yarn-Mistral-7b-64k" in args.model[0][0]:
        print("use yarn mistral")
        from attention.configuration_mistral import MistralConfig
    else:
        print("use transformers mistral")
        from transformers import MistralConfig
    config_cls = MistralConfig
    model_cls = MistralForCausalLM
    
    print("aggressive_mem_causal_lm", args.aggressive_mem_causal_lm)
    if args.aggressive_mem_causal_lm:
        transformers.models.mistral.modeling_mistral.MistralForCausalLM.forward = forward_mistral_for_causal_lm
        
    print("aggressive-mem-decoder", args.aggressive_mem_decoder)
    if args.aggressive_mem_decoder:
        transformers.models.mistral.modeling_mistral.MistralDecoderLayer.forward = forward_mistral_decoder_layer
        print('Replaced forward functions.')
        
    print("flash_attn", args.flash_attn)
    if args.flash_attn:
        print("use replace flash attn")
        # replace_mistral_attn(use_flash_attn=True, use_full=True, inference=True)
        replace_mistral_attn(use_flash_attn=True, use_full=True, inference=True, aggressive_memory=args.aggressive_mem_attn)
            # replace_llama_attn(use_flash_attn=True, use_full=True, inference=True)
        # else:
        #     raise ValueError("name not in mistral")
    
    model_name = model

    config = transformers.AutoConfig.from_pretrained(
    # config = config_cls.from_pretrained(
        model_name,
        cache_dir=args.cache_dir,
    )
    
    if args.sliding_window_attention:
        config.sliding_window = args.sliding_window_attention
        
    if "MistralLite" in args.model[0][0]:
        config.sliding_window = 16384
    scaling_factor = float(args.factor)
    
    print(model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        cache_dir=args.cache_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        # trust_remote_code=True if "Yarn" in model_name else False
    )   
    
    
    # load rope_para:
    # ft: 4k 8k 256k 512k 1024k 
    if args.finetuned and args.method == "longrope":
        print("args.finetuned", args.finetuned, "use rope_scale.pt")
        if args.max_tokens != None:
            seq_len = (args.max_tokens + 1023) // 1024
            seq_range = [0, 4, 8, 16, 32, 128, 256, 512, 1024, 2048, 10000]
            for i in range(len(seq_range)-1):
                if seq_range[i] <= seq_len <= seq_range[i+1]:   
                    seq_len = seq_range[i+1]
                    break
            if config.model_type == "mistral": 
                model_type = "mis"
            else:
                raise ValueError("model_type is not mistral")  
            ft_model_len = (config.sliding_window + 1023) // 1024
            print("config.sliding_window", config.sliding_window)
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
            if para_key in ['128k_mis_256k', '16k_mis_128k', '32k_mis_128k', '16k_mis_256k','32k_mis_256k',]:
                para_key = 'ft_mis_256k'
                
            rope_rescale = torch.load("./evaluation/rope_rescale-new.pt")
            # dict_keys(['1024k_la2_128k', '1024k_mis_256k', '2048k_mis_128k', '256k_mis_128k', '512k_mis_128k', '1024k_la2_256k', '2048k_la2_128k', '2048k_mis_256k', '512k_la2_128k', '512k_mis_256k', '1024k_mis_128k', '2048k_la2_256k', '256k_la2_128k', '512k_la2_256k', '16k_la2_128k', '8k_la2_128k', '4k_la2_256k', '8k_mis_128k', '32k_la2_128k', '16k_la2_256k', '8k_la2_256k', '4k_mis_256k', '4k_la2_128k', '32k_la2_256k', '4k_mis_128k', '8k_mis_256k', 'ft_la2_128k', 'ft_la2_256k', 'ft_mis_128k'])

            lambda_1 = rope_rescale[para_key]
        else:
            raise ValueError("args.max_tokens == None")  
    elif args.method == "longrope" and not args.finetuned:
        print("args.finetuned", args.finetuned, "Not use rope_scale.pt")
        # use base scale
        lambda_1 = np.full((32, 64), 1.0)
    else:
        print("args.finetuned", args.finetuned, "Not use rope_scale.pt")
        lambda_1 = np.full((32, 64), 1.0)
    
    
    if args.method == "yarn":
        print("\n--use ", args.method)
        from rope.MistralYaRNScaledRotaryEmbedding import MistralYaRNScaledRotaryEmbedding
        print("args.finetuned", args.finetuned)
        for each in model.model.layers:
            each.self_attn.rotary_emb = MistralYaRNScaledRotaryEmbedding(
                each.self_attn.head_dim, 
                scale=scaling_factor,
                original_max_position_embeddings=args.original_max_position_embeddings, 
                finetuned=args.finetuned, 
                device=each.self_attn.rotary_emb.inv_freq.device,
            )
    
    elif args.method == "dy_yarn":
        print("--use ", args.method)
        from rope.LlamaDynamicYaRNScaledRotaryEmbedding import LlamaDynamicYaRNScaledRotaryEmbedding
        print("args.finetuned", args.finetuned)
        for each in model.model.layers:
            each.self_attn.rotary_emb = LlamaDynamicYaRNScaledRotaryEmbedding(
                each.self_attn.head_dim,
                max_position_embeddings=args.max_position_embeddings,
                original_max_position_embeddings=args.original_max_position_embeddings, 
                finetuned=args.finetuned, 
                device=each.self_attn.rotary_emb.inv_freq.device,
                beta_fast=128
            ) 
    if args.method == "longrope":
        print("--use ", args.method)
        
        from rope.LlamaLongRoPEScaledRotaryEmbedding import LlamaLongRoPEScaledRotaryEmbedding
        print("args.finetuned", args.finetuned)
        
        assert lambda_1.shape == (32, 64), f"lambda_1 shape error {lambda_1.shape}"
        
        layer = 0
        for each in model.model.layers:
            each.self_attn.rotary_emb = LlamaLongRoPEScaledRotaryEmbedding(
                each.self_attn.head_dim, 
                scale=scaling_factor,
                original_max_position_embeddings=args.original_max_position_embeddings, 
                finetuned=args.finetuned, 
                device=each.self_attn.rotary_emb.inv_freq.device,
                lambda_1=lambda_1[layer, :],
                # tmps=args.tmps
            ) 
            layer += 1
    
    elif args.method == "longrope_start":
        print("--use ", args.method)
        from rope.LlamaLongRoPEScaledStartTokenRotaryEmbedding import LlamaLongRoPEScaledStartTokenRotaryEmbedding
        print("args.finetuned", args.finetuned)
        
        lambda_1 = np.loadtxt(open(args.longrope_para, "rb"), delimiter=",", skiprows=0)
        
        # assert lambda_1.shape == (32, 64), f"lambda_1 shape error {lambda_1.shape}"
        
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
            each.self_attn.rotary_emb = LlamaLongRoPEScaledStartTokenRotaryEmbedding(
                each.self_attn.head_dim, 
                scale=scaling_factor,
                original_max_position_embeddings=args.original_max_position_embeddings, 
                finetuned=args.finetuned, 
                device=each.self_attn.rotary_emb.inv_freq.device,
                lambda_1=lambda_1[layer, :],
                # tmps=args.tmps,
                start_token=args.stream,
                cos_sin_origin=cos_sin_origin
            ) 
            layer += 1
            
    elif args.method == "dy_longrope":
        print("--use ", args.method)
        from rope.LlamaDynamicLongRoPEScaledRotaryEmbedding import LlamaDynamicLongRoPEScaledRotaryEmbedding
        print("args.finetuned", args.finetuned)
        
        # lambda_1 = np.loadtxt(open(args.longrope_para, "rb"), delimiter=",", skiprows=0)
        
        assert lambda_1.shape == (32, 64), f"lambda_1 shape error {lambda_1.shape}"
        
        layer = 0
        for each in model.model.layers:
            each.self_attn.rotary_emb = LlamaDynamicLongRoPEScaledRotaryEmbedding(
                each.self_attn.head_dim, 
                scale=scaling_factor,
                original_max_position_embeddings=args.original_max_position_embeddings, 
                finetuned=args.finetuned, 
                device=each.self_attn.rotary_emb.inv_freq.device,
                lambda_1=lambda_1[layer, :],
                # tmps=args.tmps
            ) 
            layer += 1

    return model, lambda_1


# def add_args(parser: ArgumentParser):
    
#     parser.add_argument("--max-position-embeddings", type=int)
#     parser.add_argument("--original-max-position-embeddings", type=int)
    
#     parser.add_argument("--cache_dir", type=str)
#     parser.add_argument("--flash_attn", action="store_true")
#     parser.add_argument("--method", type=str, default="pi")
#     parser.add_argument("--longrope_para", type=str, default="./evolution/dim_mono/result_alpha/dim_mono_8192_result.csv")
#     parser.add_argument("--tmps", type=str, default="su", help='')
#     parser.add_argument("--factor", type=float)
#     parser.add_argument("--finetuned", action="store_true")
    
#     parser.add_argument("--stream", type=int, default=0)
#     parser.add_argument("--peft-model", type=str)
#     parser.add_argument("--use_cache", action="store_true")
    
#     return parser



def load_model_and_apply_patches_mistral(model, args):
    print(args)
    return load_model(model, args)
