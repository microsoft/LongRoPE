import os
import sys
import torch
import logging
from typing import List, Union

import transformers
from .huggingface import AutoCausalLM

sys.path.append(os.path.join(os.path.split(__file__)[0], os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir))
from rope import load_model


logger = logging.getLogger(__file__)
TokenSequence = Union[List[int], torch.LongTensor, torch.Tensor, transformers.BatchEncoding]


class LongRoPEAutoLM(AutoCausalLM):
    AUTO_TOKENIZER_CLASS = transformers.AutoTokenizer

    def __init__(self, model: str, batch_size: int = 1, **model_args):
        super().__init__(pretrained=model, batch_size=batch_size)

        self.tokenizer = self.AUTO_TOKENIZER_CLASS.from_pretrained(
            model, model_max_length=sys.maxsize, trust_remote_code=True, use_fast=True)

        self.tokenizer.pad_token = self.tokenizer.eos_token

        torch.cuda.empty_cache()

        logger.info(f"Loading model: {model}")
        rope_method = model_args.get('rope_method', 'none')
        if rope_method.startswith('longrope'):
            rope_params = {
                'longrope_params_path': model_args.get('longrope_params'),
                'longrope_scaling_policy': model_args.get('longrope_scaling_policy', 'su'),
            }
        else:
            rope_params = None
        max_position_embeddings = model_args.get('max_position_embeddings')
        dtype = getattr(torch, model_args.get('dtype', 'float16'))
        self.model = load_model(
            model_name_or_path=model,
            rope_method=rope_method,
            max_position_embeddings=max_position_embeddings,
            rope_params=rope_params,
            cache_dir=model_args.get('cache_dir', None),
            attn_implementation=model_args.get('attn_implementation', 'eager'),
            attn_sliding_window=model_args.get('attn_sliding_window', None),
            torch_dtype=dtype,
            save_memory=False,
            device_map='auto',
        )

        self.model.eval()
        self.model.cuda()
