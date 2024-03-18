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
from attention.llama_attn_replace import replace_llama_attn, forward_llama_for_causal_lm, forward_llama_model, forward_llama_decoder_layer

import math
import numpy as np

MODEL_LAYER = 0
MODEL_DIM = 0

def load_model(model, args):

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
    # situation
    # 1. non ft + use init_para
    # 2. ft + use init_para == search twice
    
    
    # ft: 4k 8k 256k 512k 1024k 
    if args.method == "longrope":
        if args.finetuned and not args.search_twice:
            print("Use defaut longrope para: rope_scale-new.pt")
            if args.max_tokens != None:
                seq_len = (args.max_tokens + 1023) // 1024
                seq_range = [0, 4, 8, 16, 32, 128, 256, 512, 1024, 2048, 10000]
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
                    
                print("args.max_tokens", args.max_tokens, "para_key", para_key)
                rope_rescale = torch.load("./evaluation/rope_rescale-new.pt")
                
                lambda_1 = rope_rescale[para_key]
            
            else:
                raise ValueError("args.max_tokens == None")  
        
        else:
            print("Use input longrope para")
            if args.longrope_para != None:
                # load from .csv/.pt
                if ".csv" in args.longrope_para:
                    lambda_1 = np.loadtxt(open(args.longrope_init_para, "rb"), delimiter=",", skiprows=0)
                elif ".pt" in args.longrope_para:
                    lambda_1 = torch.load(args.longrope_init_para)
                else:
                    raise f"file type not support: {args.longrope_para}"
            else:
                # use base scale
                lambda_1 = np.full((32, 64), 1.0)
    else:
        # use base scale
        lambda_1 = np.full((32, 64), 1.0)
        
    print("lambda_1 in model load ......", lambda_1)
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
    elif args.method == "longrope":
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
        
        assert lambda_1.shape == (32, 64), f"lambda_1 shape error {lambda_1.shape}"
        
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
        
        seq_len = args.max_tokens
        tmp_device = "cpu"
        rotary_emb_origin = LlamaRotaryEmbedding(dim=model.model.layers[0].self_attn.head_dim, max_position_embeddings=seq_len, device=tmp_device)
        input_x = torch.zeros((1,),dtype=torch.float16, device=tmp_device)
        cos_sin_origin = rotary_emb_origin.forward(x=input_x, seq_len=seq_len)
        # cos_sin_origin=None
        
        layer = 0
        print("start_token", args.start_token)
        for each in model.model.layers:
            each.self_attn.rotary_emb = LlamaLongRoPEScaledStartTokenRotaryEmbedding(
                each.self_attn.head_dim, 
                scale=scaling_factor,
                original_max_position_embeddings=args.original_max_position_embeddings, 
                finetuned=args.finetuned, 
                device=each.self_attn.rotary_emb.inv_freq.device,
                lambda_1=lambda_1[layer, :],
                # tmps=args.tmps,
                start_token=args.start_token,
                cos_sin_origin=cos_sin_origin
            ) 
            layer += 1
        
    elif args.method == "dy_longrope":
        print("--use ", args.method)
        from rope.LlamaDynamicLongRoPEScaledRotaryEmbedding import LlamaDynamicLongRoPEScaledRotaryEmbedding
        print("args.finetuned", args.finetuned)
        
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
    parser.add_argument("-m", "--model", action="append", nargs="+")
    
    parser.add_argument("--max_position_embeddings", type=int)
    parser.add_argument("--original_max_position_embeddings", type=int, default=4096)
    
    parser.add_argument("--cache_dir", type=str)
    parser.add_argument("--method", type=str, default="pi", 
                        choices=["pi", "ntk", "dy_ntk", "yarn", "dy_yarn", "longrope", "dy_longrope", "longrope_start"])
    
    # search eval
    parser.add_argument("--longrope_para", type=str, default=None)
    parser.add_argument("--search_twice", action="store_true")
    
    parser.add_argument("--factor", type=float)
    parser.add_argument("--finetuned", action="store_true")
    
    # accelerate
    parser.add_argument("--flash_attn", action="store_true")
    parser.add_argument("--aggressive_mem_decoder", action="store_true")
    parser.add_argument("--aggressive_mem_causal_lm", action="store_true")
    parser.add_argument("--aggressive_mem_attn", action="store_true")
    
    parser.add_argument("--start_token", type=int, default=0)
    parser.add_argument("--peft_model", type=str)
    
    
    # mistral max context window
    parser.add_argument("--sliding_window_attention", type=int)
    
    
    # use KV cache
    parser.add_argument("--use_cache", action="store_true")
    return parser



def load_model_and_apply_patches(model, args):

    return load_model(model, args)
