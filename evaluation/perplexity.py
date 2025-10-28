# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import gc
import sys
import logging
import argparse
import warnings

import torch
import datasets
import transformers
from tqdm import tqdm

sys.path.append(os.path.join(os.path.split(__file__)[0], os.path.pardir))
from rope import load_model
from utils.save_memory import replace_methods


logger = logging.getLogger(__file__)


def compute_perplexity(
    dataset: datasets.IterableDatasetDict,
    model: transformers.AutoModelForCausalLM,
    tokenizer: transformers.AutoTokenizer,
    add_start_token: bool = True,
    num_tokens: int = None,
    sliding_window: int = 256,
    max_sliding_count: int = 10,
    truncate: bool = False,
    use_cache: bool = False,
    save_memory: bool = False,
    device: torch.device = None,
):
    r"""Compute "sliding window" perplexity on a dataset. Validated against the calculations reported in arXiv 2306.15595  """

    if save_memory:
        replace_methods(model)

    if add_start_token:
        # leave room for <BOS> token to be added:
        assert tokenizer.bos_token is not None, \
            "Input model must already have a BOS token if using add_start_token=True. " \
            "Please use a different model, or set add_start_token=False"
        max_tokenized_len = num_tokens - 1
    else:
        max_tokenized_len = num_tokens

    encoded_texts = dataset["input_ids"]
    if "cus_labels" in dataset:
        cus_labels = dataset["cus_labels"]
    else:
        cus_labels = None

    if num_tokens and truncate:
        encoded_texts = [x[:max_tokenized_len] for x in encoded_texts]
        cus_labels = [x[:max_tokenized_len] for x in cus_labels] if cus_labels is not None else cus_labels
        sliding_window = max_tokenized_len + 1 if add_start_token else max_tokenized_len

    pbar = tqdm(total=len(encoded_texts), disable=logger.level <= logging.INFO)

    nlls = []
    for i, encoded_text in enumerate(encoded_texts):
        inputs = torch.tensor([encoded_text], device=device)
        if cus_labels is not None:
            labels = torch.tensor([cus_labels[i]], device=device)
        else:
            labels = torch.tensor([encoded_text], device=device)
        seq_len = labels.size(1)
        seq_len = min(seq_len, max_sliding_count * sliding_window)

        prev_end_loc = 0

        for begin_loc in range(0, seq_len, sliding_window):

            end_loc = min(begin_loc + max_tokenized_len, seq_len)
            trg_len = end_loc - prev_end_loc
            input_ids = inputs[:, begin_loc:end_loc]
            target_ids = labels[:, begin_loc:end_loc]

            if add_start_token:
                bos_tokens_tensor = torch.tensor([[tokenizer.bos_token_id]] * input_ids.size(0), device=device)
                input_ids = torch.cat([bos_tokens_tensor, input_ids], dim=1)
                target_ids = torch.cat([bos_tokens_tensor, target_ids], dim=1)

            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                with torch.cuda.device(device):
                    outputs = model(input_ids, labels=target_ids, use_cache=use_cache)
                    neg_log_likelihood = outputs.loss
                    outputs = None
                    input_ids = None
                    target_ids = None
                    gc.collect()
                    torch.cuda.empty_cache()

            nlls.append(neg_log_likelihood)

            ppl = float(torch.exp(torch.stack(nlls).mean()).float().cpu())
            pbar.set_postfix(ppl=ppl)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        pbar.update(1)

    return float(torch.exp(torch.stack(nlls).mean()).float().cpu())


def main(args):
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading tokenized dataset: {args.tokenized}")
    dataset = datasets.load_from_disk(args.tokenized)
    if args.dataset_min_tokens:
        dataset = dataset.filter(lambda x: x["tokenized_len"] >= args.dataset_min_tokens if "tokenized_len" in x else len(x["input_ids"]) >= args.dataset_min_tokens, num_proc=args.num_proc)
    if args.samples:
        dataset = dataset[:args.samples]

    logger.info(f"Loading model: {args.model}")
    rope_method = os.environ.get('ROPE_METHOD', None)
    if rope_method.startswith('longrope'):
        rope_params = {
            'longrope_params_path': os.environ['LONGROPE_RESCALE_FACTOR'],
            'longrope_scaling_policy': os.environ['LONGROPE_SCALING_POLICY'],
        }
    else:
        rope_params = None

    if args.dtype is None:
        dtype = 'auto'
    else:
        dtype = getattr(torch, args.dtype)
        torch.set_default_dtype(dtype)

    max_num_tokens = int(os.environ['TARGET_LENGTH'])
    logger.info(f"Loading model: {args.model} (max_position_embeddings={max_num_tokens})")
    model = load_model(
        model_name_or_path=args.model,
        rope_method=rope_method,
        max_position_embeddings=max_num_tokens,
        rope_params=rope_params,
        attn_implementation=args.attn_implementation,
        attn_sliding_window=args.attn_sliding_window,
        torch_dtype=dtype,
        save_memory=args.save_memory,
        device_map='auto',
    )

    logger.info(f"Begin Test")
    results = []
    for num_tokens in map(int, args.num_tokens.split(',')):
        ppl = compute_perplexity(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            add_start_token=tokenizer.bos_token is not None,
            num_tokens=num_tokens,
            sliding_window=args.ppl_sliding_window,
            truncate=args.truncate,
            use_cache=args.use_cache,
            device='cuda',
        )
        logger.info(f"Perplexity (num_tokens = {num_tokens}) = {ppl}")
        results.append([num_tokens, ppl])

    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write('length,ppl\n')
            f.write('\n'.join([','.join(map(str, result)) for result in results]))


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(message)s',
        datefmt='%m-%d %H:%M',
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--tokenized", type=str)
    parser.add_argument("--dataset-min-tokens", type=int, default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--num-tokens", type=str, default='4096,8192,16384')
    parser.add_argument("--ppl-sliding-window", type=int, default=256)
    parser.add_argument("--truncate", action="store_true")
    parser.add_argument("--output-file", type=str, default=None)
    parser.add_argument("--attn-implementation", type=str, default="eager")
    parser.add_argument("--attn-sliding-window", type=int, default=-1)
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--num-proc", type=int, default=4)
    parser.add_argument("--save-memory", action="store_true")
    parser.add_argument("--dtype", type=str, default=None)
    main(parser.parse_args())
