import argparse
import datasets
import gc
import sys
import torch
import warnings
from transformers import AutoTokenizer, LlamaTokenizer

try:    
    import cube 
except:
    pass

from tqdm import tqdm

import os
current_path = os.getcwd()
sys.path.append(current_path)
print(current_path)

from evaluation.model_loader_llama import *
import numpy as np

from datasets import Dataset
import time

# from scaled_rope.modeling_llama import LlamaPartTmpSerachRotaryEmbedding
import torch
import math
                
def compute_perplexity_cube(
    model, infer_fn, tokenizer, encodings, add_start_token: bool = True, device=None, \
        max_length=None, sliding_window=256, truncate=False, config=None, args=None,
):
    print("sliding_window", sliding_window)
    r"""Compute "sliding window" perplexity on a dataset. Validated against the calculations reported in arXiv 2306.15595  """
    if device is not None:
        assert device in ["gpu", "cpu",
                          "cuda"], "device should be either gpu or cpu."
        if device == "gpu":
            device = "cuda"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if add_start_token:
        # leave room for <BOS> token to be added:
        assert (
            tokenizer.bos_token is not None
        ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
        max_tokenized_len = max_length - 1
    else:
        max_tokenized_len = max_length

    encoded_texts = encodings["input_ids"]
    attn_masks = encodings["attention_mask"]

    if max_length and truncate:
        encoded_texts = [x[0:max_tokenized_len] for x in encoded_texts]
        attn_masks = [x[0:max_tokenized_len] for x in attn_masks]
        sliding_window = max_tokenized_len

    pbar = tqdm(total=len(encoded_texts))
    # for text in encoded_texts:
    #     print("--encoded_texts", len(text))
    
    nlls = []
    for encoding_index in range(0, len(encoded_texts)):
        
        labels = torch.tensor(encoded_texts[encoding_index:encoding_index+1])
        sequence_length = labels.size(1)

        prev_end_loc = 0
        
        for begin_loc in range(0, sequence_length, sliding_window):
            # skip last block
            if args.max_tokens > 256 * 1024:
                need_pad = (begin_loc + max_tokenized_len) > sequence_length
                if need_pad:
                    continue
            
            end_loc = min(begin_loc + max_tokenized_len, sequence_length)
            trg_len = end_loc - prev_end_loc
            input_ids = labels[:, begin_loc:end_loc].to(device)
            # print("input_ids", input_ids.shape)
            if add_start_token:
                bos_tokens_tensor = torch.tensor(
                    [[tokenizer.bos_token_id]] * input_ids.size(dim=0)).to(device)
                input_ids = torch.cat(
                    [bos_tokens_tensor, input_ids], dim=1)
            
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100
            
            # not compute on loss
            # input_ids = input_ids.to(torch.bfloat16)
            
            # load rope_para:
            # ft: 4k 8k 256k 512k 1024k 
            if config.model_type == "llama":
                if args.finetuned and args.method == "longrope":
                    # print("Use defaut longrope para: rope_scale-new.pt")
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
                        rope_rescale = torch.load("./evaluation/rope_rescale-new-2.pt")
                        
                        lambda_1 = rope_rescale[para_key]
                    
                    else:
                        raise ValueError("args.max_tokens == None")  
                
                elif args.method == "longrope" and not args.finetuned:
                    if args.longrope_para != None:
                        # print("Use input longrope para")
                        # load from .csv/.pt
                        if ".csv" in args.longrope_para:
                            lambda_1 = np.loadtxt(open(args.longrope_para, "rb"), delimiter=",", skiprows=0)
                        elif ".pt" in args.longrope_para:
                            lambda_1 = torch.load(args.longrope_para)
                        else:
                            raise f"file type not support: {args.longrope_para}"
                    else:
                        # print("Use base scale (1.0)")
                        lambda_1 = np.full((32, 64), 1.0)
                else:
                    # use base scale
                    lambda_1 = np.full((32, 64), 1.0)
            else:
                # load rope_para:
                # ft: 4k 8k 256k 512k 1024k 
                if args.finetuned and args.method == "longrope":
                    # print("args.finetuned", args.finetuned, "use rope_scale.pt")
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
                        # print("config.sliding_window", config.sliding_window)
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
                        
                        print("$$args.max_tokens", args.max_tokens, "para_key", para_key)
                        
                        lambda_1 = rope_rescale[para_key]
                    else:
                        raise ValueError("args.max_tokens == None")  
                elif args.method == "longrope" and not args.finetuned:
                    if args.longrope_para != None:
                        # print("Use input longrope para")
                        # load from .csv/.pt
                        if ".csv" in args.longrope_para:
                            lambda_1 = np.loadtxt(open(args.longrope_para, "rb"), delimiter=",", skiprows=0)
                        elif ".pt" in args.longrope_para:
                            lambda_1 = torch.load(args.longrope_para)
                        else:
                            raise f"file type not support: {args.longrope_para}, must in [.pt, .csv]"
                    else:
                        # print("Use base scale (1.0)")
                        lambda_1 = np.full((32, 64), 1.0)
                else:
                    # print("args.finetuned", args.finetuned, "Not use rope_scale.pt")
                    lambda_1 = np.full((32, 64), 1.0)

            if torch.distributed.get_rank() == 0:
                print("lambda_1 in model load ......", lambda_1)
 
            scaling_factor = float(args.factor)

            rep_method = [args.rope_method for _ in range(32)]
            rep_tmps = [args.rope_tmps for _ in range(32)]
            if not isinstance(lambda_1, torch.Tensor):           
                lamda_1_tensor = torch.from_numpy(lambda_1).to(torch.device('cuda'))
            else:
                lamda_1_tensor = lambda_1.to(torch.device('cuda'))
            rep_scaling_factor = [scaling_factor for _ in range(32)]
            rep_finetuned = [args.finetuned for _ in range(32)]
            rep_start_token = [args.start_token for _ in range(32)]
            if args.sliding_window_attention is None:
                rep_sliding_window_attention = [args.max_tokens for _ in range(32)]
            else:
                rep_sliding_window_attention = [args.sliding_window_attention for _ in range(32)]

            with torch.no_grad():
                # NOTE: use cube compiled model (yuanyuan)
                outputs = infer_fn[0](
                            model,
                            input_ids,
                            target_ids,
                            rep_method,
                            rep_tmps,
                            lamda_1_tensor,
                            rep_scaling_factor,
                            rep_finetuned,
                            rep_start_token,
                            rep_sliding_window_attention,
                            )
                
                neg_log_likelihood = outputs['loss']
            
            if torch.distributed.get_rank() == 0:
                print(outputs)
            # if aggressive_memory:
            outputs = None
            input_ids = None
            target_ids = None
            gc.collect()
            torch.cuda.empty_cache()

            nlls.append(neg_log_likelihood)

            ppl = float(torch.exp(torch.stack(nlls).mean()).float().cpu())
            # print(begin_loc, ppl)
            pbar.set_postfix(ppl=ppl)

            prev_end_loc = end_loc
            if end_loc == sequence_length:
                break

        pbar.update(1)

    ppl = float(torch.exp(torch.stack(nlls).mean()).float().cpu())
    return {"mean_perplexity": ppl}

def compute_perplexity(
    encodings, model, tokenizer, add_start_token: bool = True, device=None, max_length=None, sliding_window=256, truncate=False, 
    # aggressive_memory=False, 
    use_cache=False
):
    # sliding_window = sliding_window * 16
    print("sliding_window", sliding_window)
    r"""Compute "sliding window" perplexity on a dataset. Validated against the calculations reported in arXiv 2306.15595  """
    if device is not None:
        assert device in ["gpu", "cpu",
                          "cuda"], "device should be either gpu or cpu."
        if device == "gpu":
            device = "cuda"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # print("device", device)
    
    if add_start_token:
        # leave room for <BOS> token to be added:
        assert (
            tokenizer.bos_token is not None
        ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
        max_tokenized_len = max_length - 1
    else:
        max_tokenized_len = max_length

    encoded_texts = encodings["input_ids"]
    attn_masks = encodings["attention_mask"]

    if max_length and truncate:
        encoded_texts = [x[0:max_tokenized_len] for x in encoded_texts]
        attn_masks = [x[0:max_tokenized_len] for x in attn_masks]
        sliding_window = max_tokenized_len

    pbar = tqdm(total=len(encoded_texts))
    # for text in encoded_texts:
    #     print("--encoded_texts", len(text))
    
    nlls = []
    for encoding_index in range(0, len(encoded_texts)):
        
        labels = torch.tensor(encoded_texts[encoding_index:encoding_index+1])
        # print("len", labels.shape)
        seq_len = labels.size(1)

        prev_end_loc = 0
        
        for begin_loc in range(0, seq_len, sliding_window):
            # skip last block
            if max_tokenized_len > 256 * 1024:
                need_pad = (begin_loc + max_tokenized_len) > seq_len
                if need_pad:
                    continue
            
            end_loc = min(begin_loc + max_tokenized_len, seq_len)
            # print(begin_loc, end_loc)
            trg_len = end_loc - prev_end_loc
            input_ids = labels[:, begin_loc:end_loc].to(device)
            # print("input_ids", input_ids.shape)
            if add_start_token:
                bos_tokens_tensor = torch.tensor(
                    [[tokenizer.bos_token_id]] * input_ids.size(dim=0)).to(device)
                input_ids = torch.cat(
                    [bos_tokens_tensor, input_ids], dim=1)

            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100
            
            # not compute on loss
            # input_ids = input_ids.to(torch.bfloat16)
            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids,
                                use_cache=use_cache
                                )
                neg_log_likelihood = outputs.loss

            # print(outputs)

            # if aggressive_memory:
            outputs = None
            input_ids = None
            target_ids = None
            gc.collect()
            torch.cuda.empty_cache()

            nlls.append(neg_log_likelihood)

            ppl = float(torch.exp(torch.stack(nlls).mean()).float().cpu())
            # print(begin_loc, ppl)
            pbar.set_postfix(ppl=ppl)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        pbar.update(1)

    ppl = float(torch.exp(torch.stack(nlls).mean()).float().cpu())
    return {"mean_perplexity": ppl}


def main(args):
    if args.use_cube:
        cube.init()
    models = [x[0] for x in args.model]
    # tokenizer = LlamaTokenizer.from_pretrained(
    print("models", models)
    tokenizer = AutoTokenizer.from_pretrained(
        models[0]
        )
    
    tokenizer.pad_token = tokenizer.eos_token

    if args.tokenized:
        try:
            print(f"load from {args.tokenized}")
            input_texts = datasets.load_from_disk(args.tokenized)
            print("load finish")
        except:
            input_texts = datasets.load_dataset(
                args.tokenized, name=args.subset, split=args.split)
    else:
        input_texts = datasets.load_dataset(
            args.dataset, name=args.subset, split=args.split)

        def tokenize(example):
            tokenized = tokenizer(
                example[args.feature],
                add_special_tokens=False,
                padding=True,
                truncation=False,
                max_length=sys.maxsize,
                return_attention_mask=True,
            )
            example["input_ids"] = tokenized["input_ids"]
            example["attention_mask"] = tokenized["attention_mask"]
            example["tokenized_len"] = len(tokenized["input_ids"])
            return example

        input_texts = input_texts.map(tokenize)
        if args.save_tokenized:
            input_texts.save_to_disk(args.save_tokenized)
            print(f"Saved tokenized dataset to {args.save_tokenized}")
            return

    if "books" in args.tokenized:
        config = transformers.AutoConfig.from_pretrained(models[0], cache_dir=args.cache_dir)
        save_path = f"books_type_{config.model_type}_min{args.dataset_min_tokens}.pt"
        if os.path.exists(save_path):
            input_texts = torch.load(save_path)
        else:
            if args.dataset_min_tokens:
                # args.dataset_min_tokens = args.min_tokens
                input_texts = input_texts.filter(
                    lambda x: x["tokenized_len"] >= args.dataset_min_tokens)
            if args.samples:
                input_texts = input_texts[:min(args.samples, len(input_texts))]
                # input_texts.save_to_disk("./evaluation/dataset/books3/")
                if min(args.samples, len(input_texts))==0:
                    raise ValueError("Seq too long! No sample")
                else:
                    print("samples: ", args.samples, len(input_texts))
            torch.save(input_texts, save_path)
    else:   
        if args.dataset_min_tokens:
            # args.dataset_min_tokens = args.min_tokens
            input_texts = input_texts.filter(
                    lambda x: x["tokenized_len"] >= args.dataset_min_tokens)
        if args.samples:
            input_texts = input_texts[:min(args.samples, len(input_texts))]
            # input_texts.save_to_disk("./evaluation/dataset/books3/")
            if min(args.samples, len(input_texts))==0:
                raise ValueError("Seq too long! No sample")
            else:
                print("samples: ", args.samples, len(input_texts))
    

    if args.tokens_step:
        tokens = [x for x in range(
            args.min_tokens, args.max_tokens + 1, args.tokens_step)]
    else:
        tokens = [args.min_tokens]
        while args.min_tokens < args.max_tokens:
            point = tokens[-1] * 2
            if point <= args.max_tokens:
                tokens.append(point)
            else:
                break

    results = []
    for model in tqdm(models, desc="Model", leave=False):
        torch.cuda.empty_cache()

        result = []
        for max_length in tokens:
            
            if args.use_cube:
                config = transformers.AutoConfig.from_pretrained(model, cache_dir=args.cache_dir)
                if args.cube_trace:
                    if config.model_type == "mistral":
                        print(args.model[0])
                        from evaluation.model_loader_mistral_cube import load_model_and_apply_patches_mistral_ppl, update_config
                        loaded = load_model_and_apply_patches_mistral_ppl(model, config, args)
                        config = update_config(config, args)
                    elif config.model_type == "llama":
                        print(args.model[0])
                        from evaluation.model_loader_llama_cube import load_model_and_apply_patches_ppl, update_config
                        loaded = load_model_and_apply_patches_ppl(model, config, args)
                        config = update_config(config, args)
                    else:
                        raise ValueError("Model type did not support!")
                else:
                    if config.model_type == "mistral":
                        print(args.model[0])
                        from evaluation.model_loader_mistral_cube import update_config
                        config = update_config(config, args)
                    elif config.model_type == "llama":
                        print(args.model[0])
                        from evaluation.model_loader_llama_cube import update_config
                        config = update_config(config, args)
                    else:
                        raise ValueError("Model type did not support!")
                    loaded = None

                print("config: ", config)
                from evaluation.cube_api import compile_model_ppl
                if args.cube_trace:
                    loaded, infer_fn = compile_model_ppl(loaded, args, config, 1024)                    
                    return
                else:
                    loaded, infer_fn = compile_model_ppl(loaded, args, config, args.max_tokens)                    
            else:
                # print(model)
                config = transformers.AutoConfig.from_pretrained(model, cache_dir=args.cache_dir)
                print("config", config)
                if config.model_type == "mistral":
                    print(args.model[0])
                    from evaluation.model_loader_mistral import load_model_and_apply_patches_mistral
                    loaded, _ = load_model_and_apply_patches_mistral(model, args)
                elif config.model_type == "llama":
                    print(args.model[0])
                    loaded, _ = load_model_and_apply_patches(model, args)
                else:
                    raise ValueError("Model type did not support!")

                infer_fn = None
            
            if args.use_cube:
                s_time = time.time()
                ppl = compute_perplexity_cube(
                    model=loaded, infer_fn=infer_fn, tokenizer=tokenizer, encodings=input_texts,
                    add_start_token=tokenizer.bos_token is not None, max_length=max_length,
                    sliding_window=args.sliding_window, truncate=args.truncate, config=config, args=args,
                    )['mean_perplexity']
                
                print(f"{model}: {max_length}={ppl}")
                result.append(ppl)
                e_time = time.time()
                print("Time cost:", e_time - s_time)  
            else:
                s_time = time.time()
                print("args.use_cache", args.use_cache)
                ppl = compute_perplexity(
                    model=loaded, tokenizer=tokenizer, encodings=input_texts,
                    add_start_token=tokenizer.bos_token is not None, max_length=max_length,
                    sliding_window=args.sliding_window, truncate=args.truncate,
                    # aggressive_memory=args.aggressive_memory,
                    use_cache=args.use_cache
                    )['mean_perplexity']
                
                print(f"{model}: {max_length}={ppl}")
                result.append(ppl)
                e_time = time.time()
                print("Time cost:", e_time - s_time)

        result.insert(0, model)
        results.append(result)

    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write(f",{','.join([str(x) for x in tokens])}\n")
            for result in results:
                f.write(f"{','.join([str(x) for x in result])}\n")


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    parser = argparse.ArgumentParser()
    # parser.add_argument("-m", "--model", action="append", nargs="+")
    parser.add_argument("-d", "--dataset", type=str)
    parser.add_argument("-s", "--subset", type=str)
    parser.add_argument("-f", "--feature", type=str)
    
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--min_tokens", type=int, default=256)
    parser.add_argument("--dataset_min_tokens", type=int)
    parser.add_argument("--tokens_step", type=int)
    parser.add_argument("--sliding_window", type=int, default=256)
    # parser.add_argument("--context_window", type=int, default=65536)
    parser.add_argument("--truncate", action="store_true")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--samples", type=int)
    parser.add_argument("--save_tokenized", type=str)
    parser.add_argument("--tokenized", type=str)
    parser.add_argument("--output_file", type=str)
    # parser.add_argument("--original_max_position_embeddings", type=int, default=4096)
    
    # NOTE: for cube
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--use_cube", action="store_true")
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--cube_trace", action="store_true")
    parser.add_argument("--rope_method", type=str, default="s_pi")
    parser.add_argument("--rope_tmps", type=str, default="su")
    
    main(add_args(parser).parse_args())
    # main(parser.parse_args())
