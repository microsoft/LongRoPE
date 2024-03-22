import sys
import os
current_path = os.getcwd()
sys.path.append(current_path)
print(current_path)

import argparse
import datasets
import numpy as np
import sys
import torch
import warnings
import time
import logging
import gc

from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer
from tqdm import tqdm
from evaluation.model_loader_llama import *
from datasets import Dataset
from evaluation.perplexity import compute_perplexity


DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"
get_time = False

import math

def get_unique_filename(filename):
    counter = 1
    while os.path.exists(filename):
        base, ext = os.path.splitext(filename)
        print("exist")
        filename = f"{base}_{counter}{ext}"
        counter += 1
    return filename

def log(text):
    try:
        logging.info(text)
        print(text)  
    except IOError as e:
        if e.errno == errno.EPIPE:
            pass
        
# select proper config from model
MODEL_LAYER = 0
MODEL_DIM = 0


# # Evolution
# class History:
#     def __init__(self) -> None:
#         self.alpha_list = []
#         self.ppl_list = []
        
#     def clear(self):
#         self.alpha_list = []
#         self.ppl_list = []

#     def __contains__(self, other):
#         if isinstance(other, np.ndarray):
#             for item in self.alpha_list:
#                 if np.array_equal(item, other):
#                     return True
#             return False

#     def add(self, indv):
#         self.alpha_list.append(indv[0])
#         self.ppl_list.append(indv[1])


import random
import json

def main(args):
    models = [x[0] for x in args.model]
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
        # max_input_len = MODEL_LAYER * 1024
        
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
        input_texts = input_texts.filter(
                lambda x: x["tokenized_len"] >= args.dataset_min_tokens)
    if args.samples:
        input_texts = input_texts[:min(args.samples, len(input_texts))]


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
        start_time = time.time()
        result = []
        for max_length in tokens:
            
            config = transformers.AutoConfig.from_pretrained(model, cache_dir=args.cache_dir)
            print("config", config)
            if config.model_type == "mistral":
                print(args.model[0])
                from evaluation.model_loader_mistral import load_model_and_apply_patches_mistral
                loaded, lambda_1 = load_model_and_apply_patches_mistral(model, args)
            elif config.model_type == "llama":
                print(args.model[0])
                loaded, lambda_1 = load_model_and_apply_patches(model, args)
            else:
                raise ValueError("Model type did not support!")
            
            s_time = time.time()
            
            MODEL_LAYER = config.num_hidden_layers
            MODEL_DIM = int(config.hidden_size / config.num_attention_heads)
            print("MODEL_LAYER, MODEL_DIM", MODEL_LAYER, MODEL_DIM)
            config_compute = [loaded, tokenizer, input_texts, tokenizer.bos_token]
            
            # set seed
            seed = 42
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)
     
            # load evo para from json
            json_path = f"./evolution/{args.longrope_method}/parameters_dict.json"
            with open(json_path, 'r') as json_file:
                loaded_parameters = json.load(json_file)

            # load init solution from csv:
            if args.factor == 1.0:
                init_alpha = np.full((MODEL_LAYER, MODEL_DIM//2), 1.0     )
            else:
                init_alpha = np.loadtxt(open(args.longrope_init_para, "rb"), delimiter=",", skiprows=0)
            
            scale = max_length / args.original_max_position_embeddings
                   
            if args.longrope_method == "dim_piece_mono":
                
                from evolution.dim_piece_mono.dim_piece_mono import DimPieceMonoGeneticAlgorithm
                assert init_alpha.shape == (MODEL_DIM//2 + 2,), f"init_alpha shape error {init_alpha.shape}"
                
                genetic_algorithm = DimPieceMonoGeneticAlgorithm(
                    args, config_compute, max_length, 
                    loaded_parameters = loaded_parameters,
                    verbose = True, 
                    init_alpha=init_alpha,
                    lambda_1=lambda_1)

                best_result = genetic_algorithm.run_genetic_algorithm()
                # best_result = [np.zeros((66)), 0]
                
                if best_result[0].shape != (MODEL_LAYER, MODEL_DIM//2):
                    new_alpha = best_result[0][2:]
                    assert new_alpha.shape == (MODEL_DIM//2,)
                    best_result[0] = np.tile(new_alpha, (MODEL_LAYER, 1))
              
            elif args.longrope_method == "dim_mono":
                from evolution.dim_mono.dim_mono import DimMonoGeneticAlgorithm
                if init_alpha.shape == (MODEL_LAYER, MODEL_DIM//2):
                    init_alpha = init_alpha[0,:]
                assert init_alpha.shape == (MODEL_DIM//2,), f"init_alpha shape error {init_alpha.shape}"
                
                genetic_algorithm = DimMonoGeneticAlgorithm(
                    args, config_compute, max_length, 
                    loaded_parameters = loaded_parameters,
                    verbose = True, 
                    init_alpha=init_alpha,
                    lambda_1=lambda_1)

                best_result = genetic_algorithm.run_genetic_algorithm()

                if best_result[0].shape != (MODEL_LAYER, MODEL_DIM//2):
                    best_result[0] = np.tile(best_result[0], (MODEL_LAYER, 1))
            
            elif args.longrope_method == "dim_mono_n":
                from evolution.dim_mono_n.dim_mono_n import DimMonoNGeneticAlgorithm
                assert init_alpha.shape == (MODEL_DIM//2+1,), f"init_alpha shape error {init_alpha.shape}"
                
                genetic_algorithm = DimMonoNGeneticAlgorithm(
                    args, config_compute, max_length, 
                    loaded_parameters = loaded_parameters,
                    verbose = True, 
                    init_alpha=init_alpha,
                    lambda_1=lambda_1)

                best_result = genetic_algorithm.run_genetic_algorithm()

                if best_result[0].shape != (MODEL_LAYER, MODEL_DIM//2):
                    best_result[0] = np.tile(best_result[0][1:], (MODEL_LAYER, 1))
            
            # save result
            max_time_budget = int(loaded_parameters["evo_scale"] * loaded_parameters["max_time_budget"])
            
            result_filename = f"./evolution/search_result/final-{args.longrope_method}-{max_length}-it-{max_time_budget}.csv"
            save_name = get_unique_filename(result_filename)
            
            assert best_result[0].shape == (MODEL_LAYER, MODEL_DIM//2), f"best_result.shape error{best_result[0].shape}"
            if args.search_twice:
                print("lambda_1.shape", lambda_1.shape)
                print("best_result[0].shape", best_result[0].shape)
                best_result[0] *= lambda_1
                
            np.savetxt(save_name, best_result[0], delimiter=',',fmt='%16.16f' )
            
            result_filename = f"./evolution/search_result/final-{args.longrope_method}-{max_length}-it-{max_time_budget}.pt"
            save_name = get_unique_filename(result_filename)
            
            torch.save(best_result[0], save_name)
            
            
        end_time = time.time()
        ss_time = (end_time - start_time)
        print('time:%.3fs'%ss_time)
        log("*" * 100)
        log(f'Total Time: {ss_time}')
        
if __name__ == "__main__":
    warnings.simplefilter("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", action="append", nargs="+")
    parser.add_argument("-d", "--dataset", type=str, default="tau/scrolls")
    parser.add_argument("-s", "--subset", type=str, default="gov_report")
    parser.add_argument("-f", "--feature", type=str, default="input")
    parser.add_argument("--max_tokens", type=int, default=8000)
    parser.add_argument("--min_tokens", type=int, default=200)
    parser.add_argument("--tokens_step", type=int, default=200)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--output_file", type=str)

    parser.add_argument("--recovery", type=str, default=None, help="Path to recovery file")
    
    parser.add_argument("--longrope_init_para", type=str, default="./evolution/dim_mono/init_alpha/dim_mono_8192.csv")

    parser.add_argument("--longrope_method", type=str, default="dim_mono")
    parser.add_argument("--sliding_window", type=int, default=256)
    # parser.add_argument("--original_max_position_embeddings", type=int)
    
    parser.add_argument("--tokenized", type=str)
    parser.add_argument("--dataset_min_tokens", type=int)

    # parser.add_argument("--search_twice", action="store_true")
    parser.add_argument("--truncate", action="store_true")

    parser.add_argument("--sliding_window_attention", type=int)
    main(add_args(parser).parse_args())
