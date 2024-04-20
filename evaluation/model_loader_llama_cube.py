import torch
from argparse import ArgumentParser
import transformers
import os
import sys
import numpy as np
current_path = os.getcwd()
sys.path.append(current_path)
print(current_path)

def load_model(model, config, args):

    print("config", args.model[0])

    from evaluation.model_executor.long_seq_models.llama2.llama2 import LlamaForCausalLM
    model_cls = LlamaForCausalLM

    model_name = model

    scaling_factor = float(args.factor)

    # NOTE: Change base model loda function (yuanyuan)
    # return None
    local_rank = torch.distributed.get_rank()  
    if torch.cuda.is_available():  
        device = torch.device(f'cuda:{local_rank}')  
    print(f"local_rank: {local_rank}, device: {device}") 
    
    from evaluation.model_executor.utils import _set_default_torch_dtype
    torch_dtype = config.torch_dtype
    # torch_dtype = torch.float16
    print("apply_dtype", torch_dtype)
    with _set_default_torch_dtype(torch_dtype):
        if args.cpu:
            print("use cpu for compile")
            model = model_cls(config).eval()
        else:
            with torch.device(f'cuda:{local_rank}'):
                model = model_cls(config).eval()
    model.load_weight(model_name_or_path=model_name,
                    cache_dir=args.cache_dir)
    model = model.to(torch_dtype)
    
    for param in model.parameters():
        assert param.dtype == torch_dtype, f"param.dtype {param.dtype}"
        
    print("--use ", args.method)
    from rope.CubeLlamaDynamicScaledRotaryEmbedding import CubeLlamaDynamicScaledRotaryEmbedding
    print("args.finetuned", args.finetuned)
    
    # lambda_1 = np.loadtxt(open(args.s_pi_para, "rb"), delimiter=",", skiprows=0)
    # if args.s_pi_twice_para != None:
    #     lambda_twice = np.loadtxt(open(args.s_pi_twice_para, "rb"), delimiter=",", skiprows=0)
    #     if lambda_twice.shape == (64,):
    #         lambda_twice = np.tile(lambda_twice,(32,1))
    #     assert lambda_twice.shape == (32, 64), f"lambda_twice shape error {lambda_twice.shape}"
    
    #     lambda_1 = lambda_1 * lambda_twice
    # assert lambda_1.shape == (32, 64), f"lambda_1 shape error {lambda_1.shape}"
        
    for idx, each in enumerate(model.model.layers):
        each.self_attn.rotary_emb = CubeLlamaDynamicScaledRotaryEmbedding(
            dim=each.self_attn.head_dim, 
            scale=scaling_factor,
            max_position_embeddings=args.max_position_embeddings,
            original_max_position_embeddings=args.original_max_position_embeddings,
            dtype=torch_dtype,
        ) 
                
    return model, config

def load_model_ppl(model, config, args):

    print("config", args.model[0])

    from evaluation.model_executor.long_seq_models_ppl.llama2.llama2 import LlamaForCausalLM
    model_cls = LlamaForCausalLM

    model_name = model

    scaling_factor = float(args.factor)

    # NOTE: Change base model loda function (yuanyuan)
    # return None
    local_rank = torch.distributed.get_rank()  
    if torch.cuda.is_available():  
        device = torch.device(f'cuda:{local_rank}')  
    print(f"local_rank: {local_rank}, device: {device}") 
    
    from evaluation.model_executor.utils import _set_default_torch_dtype
    torch_dtype = config.torch_dtype
    # torch_dtype = torch.float16
    print("apply_dtype", torch_dtype)
    with _set_default_torch_dtype(torch_dtype):
        if args.cpu:
            print("use cpu for compile")
            model = model_cls(config).eval()
        else:
            with torch.device(f'cuda:{local_rank}'):
                model = model_cls(config).eval()
    model.load_weight(model_name_or_path=model_name,
                    cache_dir=args.cache_dir)
    model = model.to(torch_dtype)
    
    for param in model.parameters():
        assert param.dtype == torch_dtype, f"param.dtype {param.dtype}"
        
    print("--use ", args.method)
    from rope.CubeLlamaDynamicScaledRotaryEmbedding import CubeLlamaDynamicScaledRotaryEmbedding
    print("args.finetuned", args.finetuned)
    
    # lambda_1 = np.loadtxt(open(args.s_pi_para, "rb"), delimiter=",", skiprows=0)
    # if args.s_pi_twice_para != None:
    #     lambda_twice = np.loadtxt(open(args.s_pi_twice_para, "rb"), delimiter=",", skiprows=0)
    #     if lambda_twice.shape == (64,):
    #         lambda_twice = np.tile(lambda_twice,(32,1))
    #     assert lambda_twice.shape == (32, 64), f"lambda_twice shape error {lambda_twice.shape}"
    
    #     lambda_1 = lambda_1 * lambda_twice
    # assert lambda_1.shape == (32, 64), f"lambda_1 shape error {lambda_1.shape}"
        
    for idx, each in enumerate(model.model.layers):
        each.self_attn.rotary_emb = CubeLlamaDynamicScaledRotaryEmbedding(
            dim=each.self_attn.head_dim, 
            scale=scaling_factor,
            # max_position_embeddings=args.max_position_embeddings,
            original_max_position_embeddings=args.original_max_position_embeddings, 
            dtype=torch_dtype,
        ) 
                
    return model

def update_config(config, args):
    if args.sliding_window_attention:
        config.sliding_window = args.sliding_window_attention
    scaling_factor = float(args.factor)
    
    if args.method == "pi":
        print("--use ", args.method, scaling_factor)
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    elif args.method == "dy_ntk":
        print("--use ", args.method)
        config.rope_scaling = {"type": "dynamic", "factor": scaling_factor}
    return config
        
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

def load_model_and_apply_patches(model, config, args):
    # return apply_patches(load_model(model, args), args)
    return load_model(model, config, args)

def load_model_and_apply_patches_ppl(model, config, args):
    # return apply_patches(load_model(model, args), args)
    return load_model_ppl(model, config, args)