import argparse
import sys
from datasets import load_dataset
from transformers import AutoTokenizer, pipeline
from tqdm import tqdm

import os
current_path = os.getcwd()
sys.path.append(current_path)
print(current_path)
import transformers

from evaluation.model_loader_llama import *


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, model_max_length=sys.maxsize, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    config = transformers.AutoConfig.from_pretrained(args.model, cache_dir=args.cache_dir)
    
    print("config", config)
    if config.model_type == "mistral":
        print(args.model)
        from evaluation.model_loader_mistral import load_model_and_apply_patches_mistral
        loaded, _ = load_model_and_apply_patches_mistral(args.model, args)
    elif config.model_type == "llama":
        print(args.model)
        loaded, _ = load_model_and_apply_patches(args.model, args)
    else:
        raise ValueError("Model type did not support!")
    # model = load_model_and_apply_patches(args.model, args)
    model = loaded
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, pad_token_id=tokenizer.eos_token_id,
                    temperature=args.temperature, repetition_penalty=args.repetition_penalty,
                    top_k=args.top_k, penalty_alpha=args.penalty_alpha, do_sample=args.temperature is not None)

    while True:
        if args.input_file is None:
            prompt_text = input("> ")
        else:
            input(f"Press enter to read {args.input_file} ")
            prompt_text = open(args.input_file, encoding="utf=8").read()
        response = pipe(prompt_text, num_return_sequences=1, max_new_tokens=args.max_new_tokens)[
            0]["generated_text"][len(prompt_text):]
        print(f"< {response}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--input-file", type=str)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--repetition-penalty", type=float)
    parser.add_argument("--penalty-alpha", type=float)
    parser.add_argument("--top-k", type=int)

    parser.add_argument("--max_tokens", type=int, default=None)
    args = add_args(parser).parse_args()
    main(args)
