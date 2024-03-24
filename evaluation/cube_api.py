import cube
import argparse
import random
import re
import sys
import torch
import warnings
from transformers import AutoTokenizer
from tqdm import tqdm, trange
from tqdm.contrib import tenumerate
from typing import List
import time
import os, sys
current_path = os.getcwd()
sys.path.append(current_path)
print(current_path)
from evaluation.model_loader_llama_cube import *
from evaluation.model_loader_mistral_cube import *
            
# self.args_rope, self.model_to_test, self.infer_fn, self.config, self.enc, input_ids, max_new_tokens=32)

def generate(args, model, infer_fn, config, tokenizer, prompt_ids, pass_key=None, max_new_tokens=8):
    if torch.distributed.get_rank() == 0:
        print("begin test model")
    
    max_tokens = prompt_ids.shape[1]
    input_ids = prompt_ids.to(torch.device('cuda'))

    if torch.distributed.get_rank() == 0:
        print("input shape", input_ids.shape)
        print(tokenizer.decode( prompt_ids[0, -10:]))
        
        if pass_key:
            print("pass key: ", pass_key)
        
    # load rope_para:
    # ft: 4k 8k 256k 512k 1024k 
    if args.finetuned and args.method == "longrope":
        if torch.distributed.get_rank() == 0:
            print("args.finetuned", args.finetuned, "use rope_scale.pt")
        if max_tokens != None:
            seq_len = (max_tokens + 1023) // 1024
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
            if torch.distributed.get_rank() == 0:
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
            if para_key == '128k_mis_256k':
                para_key = 'ft_mis_256k'
            if para_key in ['16k_mis_128k', '32k_mis_128k', '16k_mis_256k', '32k_mis_256k']:
                para_key = 'ft_mis_128k'
            rope_rescale = torch.load("./evaluation/rope_rescale-new-2.pt")
            # dict_keys(['1024k_la2_128k', '1024k_mis_256k', '2048k_mis_128k', '256k_mis_128k', '512k_mis_128k', '1024k_la2_256k', '2048k_la2_128k', '2048k_mis_256k', '512k_la2_128k', '512k_mis_256k', '1024k_mis_128k', '2048k_la2_256k', '256k_la2_128k', '512k_la2_256k', '16k_la2_128k', '8k_la2_128k', '4k_la2_256k', '8k_mis_128k', '32k_la2_128k', '16k_la2_256k', '8k_la2_256k', '4k_mis_256k', '4k_la2_128k', '32k_la2_256k', '4k_mis_128k', '8k_mis_256k', 'ft_la2_128k', 'ft_la2_256k', 'ft_mis_128k'])
            if torch.distributed.get_rank() == 0:
                print("$$max_tokens", max_tokens, "para_key", para_key)
            
            lambda_1 = rope_rescale[para_key]
        else:
            raise ValueError("max_tokens == None")  
    elif args.method == "longrope" and not args.finetuned:
        if args.longrope_para != None:
            if torch.distributed.get_rank() == 0:
                print("Use input longrope para")
            # load from .csv/.pt
            if ".csv" in args.longrope_para:
                lambda_1 = np.loadtxt(open(args.longrope_para, "rb"), delimiter=",", skiprows=0)
            elif ".pt" in args.longrope_para:
                lambda_1 = torch.load(args.longrope_para)
            else:
                raise f"file type not support: {args.longrope_para}, must in [.pt, .csv]"
        else:
            if torch.distributed.get_rank() == 0:
                print("Use base scale (1.0)")
            lambda_1 = np.full((32, 64), 1.0)
    else:
        if torch.distributed.get_rank() == 0:
            print("args.finetuned", args.finetuned, "Not use rope_scale.pt")
        lambda_1 = np.full((32, 64), 1.0)
    
    scaling_factor = float(args.factor)

    max_gen_len = max_new_tokens
    if torch.distributed.get_rank() == 0:
        print("begin generation-------------------")
        
        print("lambda_1", lambda_1)
        
    if args.use_cache:
        if config.model_type == "mistral" or config.model_type == "Mistral":
            assert args.tp_size < 9, "tp_size should be no more than 8"
            past_key_values = [(torch.zeros(1, (max_tokens + max_new_tokens + 2), (8 // args.tp_size) , 128, dtype=torch.bfloat16, device=torch.device("cuda")), \
                torch.zeros(1, (max_tokens + max_new_tokens + 2), (8 // args.tp_size), 128, dtype=torch.bfloat16, device=torch.device("cuda"))) for _ in range(32)]
        else:
            past_key_values = [(torch.zeros(1, (max_tokens + max_new_tokens + 2), (32 // args.tp_size) , 128, dtype=torch.bfloat16, device=torch.device("cuda")), \
                torch.zeros(1, (max_tokens + max_new_tokens + 2), (32 // args.tp_size), 128, dtype=torch.bfloat16, device=torch.device("cuda"))) for _ in range(32)]
    else:
        past_key_values = None
    
    time_list = []
    cache_length = [0]
    res = []
    
    if args.use_warm_up:
        max_gen_len += 1
    
    for idx in range(max_gen_len):
        # CudaTimer().warmup()
        # CudaTimer(enable=True, predefined=True).start('e2e')
        rep_method = [args.rope_method for _ in range(32)]
        rep_tmps = [args.rope_tmps for _ in range(32)]  
        if isinstance(lambda_1, np.ndarray):
            lamda_1_tensor = torch.from_numpy(lambda_1).to(torch.device('cuda'))
        elif isinstance(lambda_1, torch.Tensor):
            lamda_1_tensor = lambda_1.to(torch.device('cuda'))
        else:
            raise ValueError("lambda_1 type error")
        rep_scaling_factor = [scaling_factor for _ in range(32)]
        rep_finetuned = [args.finetuned for _ in range(32)]
        rep_start_token = [args.start_token for _ in range(32)]
        length = [input_ids.shape[1]]
        if args.sliding_window_attention is None:
            rep_sliding_window_attention = [max_tokens for _ in range(32)]
        else:
            rep_sliding_window_attention = [args.sliding_window_attention for _ in range(32)]
        # else:
        #     input_ids = F.pad(input_ids, (0, max_tokens - length[0]), 'constant', 0)  
        t1 = time.time()
        with torch.no_grad():
            if args.use_cube:
                # NOTE: use cube compiled model (yuanyuan)
                outputs = infer_fn[0](
                            model,
                            input_ids,
                            past_key_values,
                            rep_method,
                            rep_tmps,
                            lamda_1_tensor,
                            rep_scaling_factor,
                            rep_finetuned,
                            rep_start_token,
                            cache_length,
                            rep_sliding_window_attention
                            )
            else:
                outputs = model(
                            input_ids,
                            past_key_values,
                            rep_method,
                            rep_tmps,
                            lamda_1_tensor,
                            rep_scaling_factor,
                            rep_finetuned,
                            rep_start_token,
                            cache_length,
                            rep_sliding_window_attention
                            )
                
        t2 = time.time()
        if idx == 0 and args.use_warm_up:
            continue
        time_list.append(t2 - t1)
        logits = outputs["logits"]

        next_token = logits.argmax(dim=-1, keepdim=True).view(-1)
        next_token_cpu = next_token.cpu()
        # if torch.distributed.get_rank() == 0:
        #     next_token_text = tokenizer.decode(next_token_cpu)
        #     print("next_token:", next_token_text, ", next_token_ids:", next_token)
        
        if args.use_cache:
            cache_length[0] += input_ids.shape[1]
            input_ids = next_token.unsqueeze(0).to("cuda")
        else:
            input_ids = torch.cat([input_ids[:, :length[0]], next_token.unsqueeze(0)], dim=-1)
        
        res.append(next_token_cpu)
        response = tokenizer.decode(torch.cat(res, dim=-1))
        
        # # early stop 2
        # next_token_text = tokenizer.decode(next_token)
        # if gen_len == 0 and len(str(next_token_text)) != 0:
        #     if torch.distributed.get_rank() == 0:
        #         print("----early stop [0]!= ' ' ")
        #     break
        # if gen_len == 1 and next_token_text not in [str(p) for p in range(1, 10)]:
        #     if torch.distributed.get_rank() == 0:
        #         print("----early stop [1] not in 1-9")
        #     break
        
        # early stop
        # try:
        #     flag = int(re.search(r'\d+', response).group())
        # except:
        #     flag = response
        # if flag == pass_key:
        #     break

    if torch.distributed.get_rank() == 0:
        print("ans:")
        print(response)
        
    if pass_key is not None:
        try:
            pass_key = int(re.search(r'\d+', response).group())
        except:
            pass_key = response
        return pass_key
    else:
        return response

def compile_model(loaded, args, config):
    if args.use_cube:
        
        trace_size = 2048
        
        input_ids = torch.full((1, trace_size), 256, dtype=torch.long)  
        print("prompt_ids shape: ", input_ids.shape)
        print("trace_size: ", trace_size)
        if config.model_type == "mistral" or config.model_type == "Mistral":
            from evaluation.model_executor.long_seq_models.Mistral.policy.spmd import PASSingle, PASMegatronTP, PASMegatronTPCache
        else:
            from evaluation.model_executor.long_seq_models.llama2.policy.spmd import PASSingle, PASMegatronTP, PASMegatronTPCache
        
        if args.tp_size > 1:
            if args.use_cache:
                policy = PASMegatronTPCache
            else:
                policy = PASMegatronTP
        else:
            policy = PASSingle
            
        # compile the whole model 
        if args.cube_trace:
            lfile = None
            sfile = "cube_graph.pt"
        else:
            lfile = "cube_graph.pt"
            sfile = None
            
        if args.use_cache:
            if config.model_type == "mistral" or config.model_type == "Mistral":
                assert args.tp_size < 9, "tp_size should be no more than 8"
                past_key_values = [(torch.zeros(1, trace_size, (8 // args.tp_size) , 128, dtype=torch.bfloat16, device=torch.device("cuda")), \
                    torch.zeros(1, trace_size, (8 // args.tp_size), 128, dtype=torch.bfloat16, device=torch.device("cuda"))) for _ in range(32)]
            else:
                past_key_values = [(torch.zeros(1, trace_size, (32 // args.tp_size) , 128, dtype=torch.bfloat16, device=torch.device("cuda")), \
                    torch.zeros(1, trace_size, (32 // args.tp_size), 128, dtype=torch.bfloat16, device=torch.device("cuda"))) for _ in range(32)]
        else:
            past_key_values = None
        method = ["pi" for _ in range(32)]
        tmps = ["non" for _ in range(32)]
        lambda_1 = np.zeros((32, 64))
        lambda_tensor = torch.from_numpy(lambda_1).to(torch.device('cuda'))
        scaling_factor = [1.0 for _ in range(32)]
        finetuned = [False for _ in range(32)]
        start_token= [0 for _ in range(32)]            
        length = [input_ids.shape[1]]
        if length[0] > trace_size:
            print(f"long length error {length[0]}")
            input_ids = input_ids[:, :trace_size]
            length = [input_ids.shape[1]]
        if args.sliding_window_attention is None:
            rep_sliding_window_attention = [length for _ in range(32)]
        else:
            rep_sliding_window_attention = [args.sliding_window_attention for _ in range(32)]
        
        @cube.compile(loaded, input_ids, past_key_values, method, tmps, lambda_tensor, scaling_factor, finetuned, start_token, length, rep_sliding_window_attention, \
            PAS=policy, load_graph_file=lfile, save_graph_file=sfile, override=True, model_dynamic_shape=True)
        def infer(model: torch.nn.Module, input_ids: torch.Tensor, \
                        past_key_values: List[tuple[torch.Tensor]], method: List[str], \
                        tmps: List[str], lamda_1: torch.Tensor, scaling_factor: List[float], \
                            finetuned: List[bool], start_token: List[int], length: List[int], sliding_window_attention: List[int]):
            with torch.no_grad():
                outputs = model(input_ids, 
                                past_key_values,
                                method,
                                tmps,
                                lamda_1,
                                scaling_factor,
                                finetuned,
                                start_token,
                                length,
                                sliding_window_attention
                                )
            return outputs

        print("Complete the compilation of the model")
        del loaded
        del input_ids
        model = cube.load_model()
        infer_fn = (infer,)

        # model = cube.load_model()
        # infer_fn = (cube.load_default_schedule(), )
        
        return model, infer_fn
    else:
        return loaded, None 

def compile_model_ppl(loaded, args, config, length):
        
    trace_size = length
    input_ids = torch.full((1, trace_size), 256, dtype=torch.long).to(torch.device('cuda'))
    print("prompt_ids shape: ", input_ids.shape)
    print("trace_size: ", trace_size)
    
    labels = input_ids.to(input_ids.device)

    if config.model_type == "mistral" or config.model_type == "Mistral":
        from evaluation.model_executor.long_seq_models_ppl.Mistral.policy.spmd import PASSingle, PASMegatronTP
    else:
        from evaluation.model_executor.long_seq_models_ppl.llama2.policy.spmd import PASSingle, PASMegatronTP
    
    if args.tp_size > 1:
        policy = PASMegatronTP
    else:
        policy = PASSingle
                    
    if args.cube_trace:
        lfile = None
        sfile = "cube_graph.pb"
    else:
        lfile = "cube_graph.pb"
        sfile = None

    method = ["pi" for _ in range(32)]
    tmps = ["non" for _ in range(32)]
    lambda_1 = np.zeros((32, 64))
    lambda_tensor = torch.from_numpy(lambda_1).to(torch.device('cuda'))
    scaling_factor = [1.0 for _ in range(32)]
    finetuned = [False for _ in range(32)]
    start_token= [0 for _ in range(32)]            
    if args.sliding_window_attention is None:
        rep_sliding_window_attention = [args.max_tokens for _ in range(32)]
    else:
        rep_sliding_window_attention = [args.sliding_window_attention for _ in range(32)]
        
    @cube.compile(loaded, input_ids, labels, method, tmps, lambda_tensor, \
        scaling_factor, finetuned, start_token, rep_sliding_window_attention, \
        PAS=policy, load_graph_file=lfile, save_graph_file=sfile, override=True, model_dynamic_shape=True)
    def infer(model: torch.nn.Module, input_ids: torch.Tensor, labels: torch.Tensor, method: List[str], \
                    tmps: List[str], lamda_1: torch.Tensor, scaling_factor: List[float], \
                        finetuned: List[bool], start_token: List[int], sliding_window_attention: List[int]):
        with torch.no_grad():
            outputs = model(input_ids, 
                            labels,
                            method,
                            tmps,
                            lamda_1,
                            scaling_factor,
                            finetuned,
                            start_token,
                            sliding_window_attention,
                            )
        return outputs

    print("Complete the compilation of the model")
    del loaded
    del input_ids
    del labels
    model = cube.load_model()
    infer_fn = (infer,)
    
    # model = cube.load_model()
    # infer_fn = (cube.load_default_schedule(), )
    
    return model, infer_fn
