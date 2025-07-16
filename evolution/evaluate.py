# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import sys
import json
import socket
import logging
import argparse
import warnings

import torch
import numpy as np
import datasets
import transformers

sys.path.append(os.path.join(os.path.split(__file__)[0], os.path.pardir))
import rope
from evaluation.perplexity import compute_perplexity


logger = logging.getLogger(__file__)


def main(args):
    buf_size = 4096
    sock = socket.socket()
    sock.connect((args.host, args.port))
    logger.info(f'Connected to Server [host={args.host}, port={args.port}]')

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading tokenized dataset: {args.tokenized}")
    dataset = datasets.load_from_disk(args.tokenized)
    if args.dataset_min_tokens:
        dataset = dataset.filter(lambda x: x["tokenized_len"] >= args.dataset_min_tokens if "tokenized_len" in x else len(x["input_ids"]) >= args.dataset_min_tokens, num_proc=args.num_proc)
    if args.samples:
        dataset = dataset[:args.samples]

    logger.info(f"Loading model: {args.model}")
    target_length = int(args.target_length)
    model_args = {
        'rope_method': None,
        'max_position_embeddings': target_length,
        'attn_implementation': args.attn_implementation,
        'attn_sliding_window': args.attn_sliding_window,
        'torch_dtype': torch.bfloat16,
        'device_map': 'auto',
        'save_memory': False,
    }
    torch.set_default_dtype(torch.bfloat16)
    model = rope.load_model(model_name_or_path=args.model, **model_args)
    sock.send(json.dumps({'model_ready': True}).encode())

    eval_args = {
        'tokenizer': tokenizer,
        'dataset': dataset,
        'add_start_token': tokenizer.bos_token is not None,
        'num_tokens': target_length,
        'sliding_window': args.ppl_sliding_window,
        'truncate': args.truncate,
        'use_cache': args.use_cache,
        'save_memory': args.save_memory,
        'device': 'cuda',
    }

    while True:
        msg: dict = json.loads(sock.recv(buf_size).decode())
        if msg.get('finalize', False):
            logger.info(f'Finalized.')
            break
        rope_args: dict = msg['rope_args']
        rope_class = getattr(rope, rope_args.pop('rope_class'))
        logger.info(f'Received RoPE arguments: {rope_args}')
        if 'rescale_factors' in rope_args:
            rope_args['rescale_factors'] = np.array(rope_args['rescale_factors'])
        rope.replace_rope(model, rope_class, rope_args)
        result = compute_perplexity(model=model, **eval_args)
        sock.send(json.dumps({'result': result}).encode())

    sock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=str)
    parser.add_argument("--host", type=str)
    parser.add_argument("--port", type=int)
    parser.add_argument("--model", type=str)
    parser.add_argument("--tokenized", type=str)
    parser.add_argument("--target-length", type=int)
    parser.add_argument("--dataset-min-tokens", type=int, default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--ppl-sliding-window", type=int, default=256)
    parser.add_argument("--truncate", action="store_true")
    parser.add_argument("--attn-implementation", type=str, default="flash_attention_2")
    parser.add_argument("--attn-sliding-window", type=int, default=None)
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--num-proc", type=int, default=4)
    parser.add_argument("--save-memory", action="store_true")
    args = parser.parse_args()

    warnings.simplefilter("ignore")
    logging.basicConfig(
        level=logging.WARNING,
        format=f'[Evaluator #{args.idx} | %(asctime)s] %(message)s',
        datefmt='%m-%d %H:%M',
    )

    main(args)
