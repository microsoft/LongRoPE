# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import sys
import logging
import argparse

import datasets
from transformers import AutoTokenizer


logger = logging.getLogger(__file__)


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    input_texts = datasets.load_dataset(
        args.dataset,
        name=args.subset,
        split=args.split,
        num_proc=args.num_proc,
        ignore_verifications=True,
        trust_remote_code=True,
    )

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

    input_texts = input_texts.map(tokenize, num_proc=args.num_proc)
    input_texts.save_to_disk(args.save_tokenized, num_proc=args.num_proc)
    logger.info(f"Saved tokenized dataset to {args.save_tokenized}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--split", type=str)
    parser.add_argument("--subset", type=str)
    parser.add_argument("--feature", type=str)
    parser.add_argument("--save-tokenized", type=str)
    parser.add_argument("--num-proc", type=int, default=4)

    main(parser.parse_args())
