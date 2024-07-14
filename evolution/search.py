# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import sys
import math
import json
import socket
import random
import logging
import argparse
import warnings
import datetime

import torch
import numpy as np
import transformers

sys.path.append(os.path.join(os.path.split(__file__)[0], os.path.pardir))
from rope import YaRNScaledRotaryEmbedding
from evolution.algorithms import Evaluator, DimMonoGeneticAlgorithm, DimPieceMonoGeneticAlgorithm


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
    evaluator: Evaluator,
    init_factors: np.ndarray,
    length_scale: float,
    rope_args: dict,
):
    logger.info(f'Begin select init factors')
    target_scale = length_scale
    best_ppl = np.inf
    best_factors = None
    best_scale = None
    while target_scale >= 1:
        logger.info(f'Try init factors at scale={target_scale}')
        tmp_rope_args = {
            'rope_class': 'LongRoPEScaledRotaryEmbedding',
            'rescale_factors': rescale(target_scale, length_scale, init_factors),
            'max_position_embeddings': int(target_scale * rope_args['original_max_position_embeddings']),
            **rope_args,
        }
        evaluator.set_rope(tmp_rope_args)
        ppl = evaluator.get_result()
        if ppl < best_ppl:
            best_ppl = ppl
            best_factors = rope_args['rescale_factors']
            best_scale = target_scale
        target_scale /= 2
    logger.info(f'Selected init factors with scale={best_scale}:\n{best_factors.tolist()}')
    return best_factors, best_scale


def main(args):
    sock = socket.socket()
    sock.bind(('localhost', 0))
    host, port = sock.getsockname()
    logger.info(f'Initialize server on host={host}, port={port}')

    mem_per_device = torch.cuda.mem_get_info()[1] / (1024 ** 3) * 0.8
    model_size = float(args.model_size_gb)
    devices_per_model = math.ceil(model_size / mem_per_device)
    if args.devices is None:
        device_list = list(range(torch.cuda.device_count()))
    else:
        device_list = list(map(int, args.devices.split(',')))
    assert len(device_list) >= devices_per_model, f'At least {devices_per_model} devices are required to load the model'
    model_count = len(device_list) // devices_per_model
    device_count = model_count * devices_per_model

    evaluators: list[Evaluator] = []
    sock.listen(device_count // devices_per_model)
    for idx, device_start in enumerate(range(0, device_count, devices_per_model)):
        device_end = device_start + devices_per_model
        evaluators.append(Evaluator(
            sock=sock,
            args={
                'idx': idx,
                'host': host,
                'port': port,
                'model': args.model,
                'tokenized': args.tokenized,
                'target-length': args.target_length,
                'dataset-min-tokens': args.dataset_min_tokens,
                'samples': args.samples,
                'ppl-sliding-window': args.ppl_sliding_window,
                'truncate': args.truncate,
                'attn-implementation': args.attn_implementation,
                'attn-sliding-window': args.attn_sliding_window,
                'use-cache': args.use_cache,
                'num-proc': args.num_proc,
                'save-memory': args.save_memory,
            },
            device_list=device_list[device_start:device_end],
        ))
    for evaluator in evaluators:
        evaluator.model_ready()

    logger.info(f"Loading model config: {args.model}")
    config = transformers.AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    head_size = config.hidden_size // config.num_attention_heads
    half_head_size = head_size // 2
    target_length = int(args.target_length)
    if hasattr(config, 'sliding_window') and config.sliding_window is not None:
        original_length = config.sliding_window
    else:
        original_length = config.max_position_embeddings
    length_scale = target_length / original_length if args.length_scale is None else args.length_scale
    rope_base = getattr(config, 'rope_embedding_base', getattr(config, 'rope_theta', None))
    if config.model_type == 'mistral' or config.model_type == 'mixtral':
        rope_model_type = 'mistral'
    else:
        if not (config.model_type == 'llama' or config.model_type == 'phi3'):
            logger.warning(f'Setting model type to llama for unrecognized model type: {config.model_type}')
        rope_model_type = 'llama'

    rope_args = {
        'dim': head_size,
        'scale': length_scale,
        'max_position_embeddings': target_length,
        'original_max_position_embeddings': original_length,
        'base': rope_base,
        'model_type': rope_model_type,
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
        emb = YaRNScaledRotaryEmbedding(**rope_args, **yarn_betas)
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
            init_factors, length_scale = select_init_factors(evaluators[0], init_factors, length_scale, rope_args)
            rope_args['max_position_embeddings'] = int(length_scale * original_length)
    assert init_factors.shape == (half_head_size, ), \
        f'Initial factors shape error: {init_factors.shape} != {(half_head_size, )}'
    logger.info(f'Initial factors: {init_factors}')

    if args.algorithm == "dim_piece_mono":
        final_factors = DimPieceMonoGeneticAlgorithm(
            evaluators=evaluators,
            scale=length_scale,
            target_length=target_length,
            hyper_params=hyper_params,
            init_factors=init_factors,
            rope_args=rope_args,
            log_json_path=os.path.join(args.output_dir, f'log-{args.timestamp}.json'),
            output_dir=args.output_dir,
            recovery=args.recovery,
        ).run_genetic_algorithm()[2:]
    elif args.algorithm == "dim_mono":
        final_factors = DimMonoGeneticAlgorithm(
            evaluators=evaluators,
            scale=length_scale,
            target_length=target_length,
            hyper_params=hyper_params,
            init_factors=init_factors,
            rope_args=rope_args,
            log_json_path=os.path.join(args.output_dir, f'log-{args.timestamp}.json'),
            output_dir=args.output_dir,
            recovery=args.recovery,
        ).run_genetic_algorithm()
    else:
        raise ValueError(f'Unsupported evolution search algorithm: {args.algorithm}')

    np.savetxt(
        os.path.join(args.output_dir, f"result_final.csv"),
        final_factors,
        delimiter='\n',
    )

    for evaluator in evaluators:
        evaluator.finalize()
    sock.close()


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
    parser.add_argument("--attn-sliding-window", type=int, default=-1)
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--num-proc", type=int, default=4)
    parser.add_argument("--hyper-params", type=str, default=None)
    parser.add_argument("--init-factors", type=str, default=None)
    parser.add_argument("--auto-rescale-init-factors", action="store_true")
    parser.add_argument("--length-scale", type=float, default=None)
    parser.add_argument("--recovery", type=str, default=None)
    parser.add_argument("--save-memory", action="store_true")
    parser.add_argument("--model-size-gb", type=float, default=14)
    parser.add_argument("--devices", type=str, default=None)
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
