import argparse
import datasets
import gc
import sys
import torch
import warnings
from transformers import AutoTokenizer, LlamaTokenizer

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
            
            end_loc = min(begin_loc + max_tokenized_len, seq_len)
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
    models = [x[0] for x in args.model]
    # tokenizer = LlamaTokenizer.from_pretrained(
    tokenizer = AutoTokenizer.from_pretrained(
        models[0], model_max_length=sys.maxsize, trust_remote_code=True, use_fast = False)
    
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
    parser.add_argument("-m", "--model", action="append", nargs="+")
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
    
    # mistral max context window
    parser.add_argument("--sliding_window_attention", type=int)
    
    
    main(add_args(parser).parse_args())
    # main(parser.parse_args())
