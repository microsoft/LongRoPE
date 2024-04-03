import os
import sys
import json
import random
import logging
import argparse
import warnings
import datetime

import torch
import datasets
import numpy as np
import transformers


sys.path.append(os.path.join(os.path.split(__file__)[0], os.path.pardir))
from rope import (
    load_model,
    replace_rope,
    LlamaYaRNScaledRotaryEmbedding,
    LlamaLongRoPEScaledRotaryEmbedding,
)
from evaluation.perplexity import compute_perplexity
from evolution.algorithms import DimMonoGeneticAlgorithm, DimPieceMonoGeneticAlgorithm


logger = logging.getLogger(__file__)


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def rescale(
    target_length_scale: float,
    origin_length_scale: float,
    factors: np.ndarray,
):
    return 1.0 + (factors - 1.0) * ((target_length_scale - 1.0) / (origin_length_scale - 1.0))


def select_init_factors(
    model: transformers.AutoModelForCausalLM,
    init_factors: np.ndarray,
    length_scale: float,
    rope_args: dict,
    eval_args: dict,
):
    logger.info(f'Begin select init factors')
    target_scale = length_scale
    best_ppl = np.inf
    best_factors = None
    best_scale = None
    while target_scale >= 1:
        logger.info(f'Try init factors at scale={target_scale}')
        rope_args['max_position_embeddings'] = int(target_scale * rope_args['original_max_position_embeddings'])
        rope_args['rescale_factors'] = rescale(target_scale, length_scale, init_factors)
        model = replace_rope(model, LlamaLongRoPEScaledRotaryEmbedding, rope_args)
        ppl = compute_perplexity(model=model, **eval_args)
        if ppl < best_ppl:
            best_ppl = ppl
            best_factors = rope_args['rescale_factors']
            best_scale = target_scale
        target_scale /= 2
    logger.info(f'Selected init factors with scale={best_scale}:\n{best_factors.tolist()}')
    return best_factors, best_scale


def main(args):
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading tokenized dataset: {args.tokenized}")
    dataset = datasets.load_from_disk(args.tokenized)
    if args.dataset_min_tokens:
        dataset = dataset.filter(lambda x: x["tokenized_len"] >= args.dataset_min_tokens, num_proc=args.num_proc)
    if args.samples:
        dataset = dataset[:args.samples]

    logger.info(f"Loading model: {args.model}")
    config = transformers.AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    layer_num = config.num_hidden_layers
    head_size = config.hidden_size // config.num_attention_heads
    half_head_size = head_size // 2
    target_length = int(args.target_length)
    original_length = config.max_position_embeddings
    length_scale = target_length / original_length if args.rope_scale is None else args.rope_scale
    model_args = {
        'rope_method': None,
        'max_position_embeddings': target_length,
        'cache_dir': args.cache_dir,
        'attn_implementation': args.attn_implementation,
        'attn_sliding_window': args.attn_sliding_window,
        'torch_dtype': torch.float16,
        'device_map': 'auto',
        'save_memory': args.save_memory,
    }
    model = load_model(model_name_or_path=args.model, **model_args)

    eval_args = {
        'tokenizer': tokenizer,
        'dataset': dataset,
        'add_start_token': tokenizer.bos_token is not None,
        'num_tokens': target_length,
        'sliding_window': args.ppl_sliding_window,
        'truncate': args.truncate,
        'use_cache': args.use_cache,
        'device': 'cuda',
    }

    rope_args = {
        'dim': head_size,
        'scale': length_scale,
        'max_position_embeddings': target_length,
        'original_max_position_embeddings': original_length,
        'device': None,
    }

    set_seed()

    if args.hyper_params is None:
        hyper_params_path = os.path.join(os.path.split(__file__)[0], 'default_hyper_params', f'{args.algorithm}.json')
    else:
        hyper_params_path = args.hyper_params
    logger.info(f'Load hyper-parameters from {hyper_params_path}')
    with open(hyper_params_path) as f:
        hyper_params = json.loads(f.read())

    if args.init_factors is None:
        logger.info(f'Generate initial factors by YaRN')
        # TODO: check YaRN settings for more models
        if args.yarn_settings == 'mistral' or config.model_type == 'mistral':
            yarn_betas = {
                'beta_fast': 128,
                'beta_slow': 2,
            }
        else:
            yarn_betas = {
                'beta_fast': 32,
                'beta_slow': 1,
            }
        emb = LlamaYaRNScaledRotaryEmbedding(**rope_args, **yarn_betas)
        inv_freq_mask = emb.inv_freq_mask
        inv_freq_interpolation = length_scale
        inv_freq_extrapolation = 1.0
        inv_freq = inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask
        init_factors = inv_freq.detach().cpu().numpy().flatten()
    else:
        # TODO: search mscale (attention temperature)
        logger.info(f'Load initial factors from {args.init_factors}')
        init_factors = np.loadtxt(open(args.init_factors, "rb"), delimiter=",", skiprows=0)
        if args.auto_rescale_init_factors:
            init_factors, length_scale = select_init_factors(model, init_factors, length_scale, rope_args, eval_args)
    assert init_factors.shape == (half_head_size, ), \
        f'Initial factors shape error: {init_factors.shape} != {(half_head_size, )}'
    logger.info(f'Initial factors: {init_factors}')

    if args.algorithm == "dim_piece_mono":
        final_factors = DimPieceMonoGeneticAlgorithm(
            model=model,
            scale=length_scale,
            target_length=target_length,
            hyper_params=hyper_params,
            init_factors=init_factors,
            log_json_path=os.path.join(args.output_dir, f'log-{args.timestamp}.json'),
            output_dir=args.output_dir,
            eval_args=eval_args,
            recovery=args.recovery,
        ).run_genetic_algorithm()[2:]
    elif args.algorithm == "dim_mono":
        final_factors = DimMonoGeneticAlgorithm(
            model=model,
            scale=length_scale,
            target_length=target_length,
            hyper_params=hyper_params,
            init_factors=init_factors,
            log_json_path=os.path.join(args.output_dir, f'log-{args.timestamp}.json'),
            output_dir=args.output_dir,
            eval_args=eval_args,
            recovery=args.recovery,
        ).run_genetic_algorithm()
    else:
        raise ValueError(f'Unsupported evolution search algorithm: {args.algorithm}')

    np.savetxt(
        os.path.join(args.output_dir, f"result_final.csv"),
        final_factors,
        delimiter='\n',
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--yarn-settings", type=str, choices=["mistral", "llama"], default="llama")
    parser.add_argument("--tokenized", type=str)
    parser.add_argument("--algorithm", type=str)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--target-length", type=int)
    parser.add_argument("--dataset-min-tokens", type=int, default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--ppl-sliding-window", type=int, default=256)
    parser.add_argument("--truncate", action="store_true")
    parser.add_argument("--attn-implementation", type=str, default="flash_attention_2")
    parser.add_argument("--attn-sliding-window", type=int, default=None)
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--num-proc", type=int, default=4)
    parser.add_argument("--hyper-params", type=str, default=None)
    parser.add_argument("--init-factors", type=str, default=None)
    parser.add_argument("--auto-rescale-init-factors", action="store_true")
    parser.add_argument("--rope-scale", type=float, default=None)
    parser.add_argument("--recovery", type=str, default=None)
    parser.add_argument("--save-memory", action="store_true")
    args = parser.parse_args()
    args.timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    warnings.simplefilter("ignore")

    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s | %(name)s | %(levelname)s]\n%(message)s\n',
        datefmt='%m-%d %H:%M',
        filename=os.path.join(args.output_dir, f'log-{args.timestamp}.txt'),
        filemode='w',
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(message)s')
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)
    logger = logging.getLogger(__file__)

    main(args)
