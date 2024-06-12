import os
import re
import sys
import random
import logging
import warnings
import argparse

import torch
from tqdm import tqdm, trange
from tqdm.contrib import tenumerate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


sys.path.append(os.path.join(os.path.split(__file__)[0], os.path.pardir))
from rope import load_model


logger = logging.getLogger(__file__)


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def generate_prompt(n_garbage):
    """Generates a text file and inserts an execute line at a random position."""
    n_garbage_prefix = random.randint(0, n_garbage)
    n_garbage_suffix = n_garbage - n_garbage_prefix

    task_description = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there."
    garbage_inf = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
    while len(garbage_inf) < n_garbage:
        garbage_inf = " ".join([garbage_inf] * 2)
    garbage_prefix = garbage_inf[:n_garbage_prefix]
    garbage_suffix = garbage_inf[:n_garbage_suffix]
    pass_key = random.randint(1, 50000)
    information_line = f"The pass key is {pass_key}. Remember it. {pass_key} is the pass key."
    final_question = "What is the pass key? The pass key is"
    lines = [
        task_description,
        garbage_prefix,
        information_line,
        garbage_suffix,
        final_question
    ]
    return "\n".join(lines), pass_key


def generate_answer(pipe, prompt_text):
    response = pipe(prompt_text, num_return_sequences=1, max_new_tokens=10)[0]["generated_text"][len(prompt_text):]

    try:
        answer = int(re.search(r'\d+', response).group())
    except:
        answer = response[:20]

    return answer


def calc_str_length(token_length, letters_per_token=3.65):
    return int(token_length * letters_per_token)


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    logger.info(f"Loading model: {args.model}")
    rope_method = os.environ.get('ROPE_METHOD', None)
    if rope_method.startswith('longrope'):
        rope_params = {
            'longrope_params_path': os.environ['LONGROPE_PARAMS'],
            'longrope_scaling_policy': os.environ['LONGROPE_SCALING_POLICY'],
        }
    else:
        rope_params = None

    if args.dtype is None:
        dtype = 'auto'
    else:
        dtype = getattr(torch, args.dtype)
        torch.set_default_dtype(dtype)

    logger.info(f"Begin Test")
    results = []
    if args.log_file:
        with open(args.log_file, 'w', encoding="utf-8") as f:
            f.write('')

    max_position_embeddings = int(os.environ['SEQ_LEN']) if args.finetuned else None
    model = load_model(
        model_name_or_path=args.model,
        rope_method=rope_method,
        max_position_embeddings=max_position_embeddings,
        rope_params=rope_params,
        cache_dir=args.cache_dir,
        attn_implementation=args.attn_implementation,
        attn_sliding_window=args.attn_sliding_window,
        torch_dtype=dtype,
        save_memory=args.save_memory,
        device_map='auto',
    )

    for target_num_tokens in map(int, args.num_tokens.split(',')):
        pipe = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=False,
        )
        set_seed()
        correct_count = 0
        str_length = calc_str_length(target_num_tokens)
        if args.log_file:
            with open(args.log_file, 'a', encoding="utf-8") as f:
                f.write('=' * 80 + '\n')
                f.write(f'[Target Length = {target_num_tokens}]\n')
        for i in trange(0, args.samples, desc="Samples", leave=False):
            prompt_text, pass_key = generate_prompt(str_length)
            actual_num_tokens = len(pipe.tokenizer.encode(prompt_text))
            answer = generate_answer(pipe, prompt_text)
            if answer == pass_key:
                correct_count += 1
            if args.log_file:
                with open(args.log_file, 'a', encoding="utf-8") as f:
                    f.write(f'[#{i:0>2d} / Length = {actual_num_tokens}]\n')
                    f.write(f'[Prompt]\n{prompt_text}\n')
                    f.write(f'[Passkey]\n{pass_key}\n')
                    f.write(f'[Answer]\n{answer}\n\n')
        accuracy = correct_count / args.samples
        logger.info(f"Pass-key (num_tokens = {target_num_tokens}) = {accuracy}")
        results.append([target_num_tokens, accuracy])

    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write('length,accuracy\n')
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
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--num-tokens", type=str, default='4096,8192,16384')
    parser.add_argument("--output-file", type=str, default=None)
    parser.add_argument("--log-file", type=str, default=None)
    parser.add_argument("--finetuned",  action="store_true")
    parser.add_argument("--attn-implementation", type=str, default="flash_attention_2")
    parser.add_argument("--attn-sliding-window", type=int, default=None)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--save-memory", action="store_true")
    parser.add_argument("--dtype", type=str, default=None)
    main(parser.parse_args())
