import os
import sys
import math
import torch
import torch.nn.functional as F
import transformers
from pathlib import Path
from typing import List, Mapping, NewType, Optional, Tuple, Union
from tqdm import tqdm

from transformers import BatchEncoding
from transformers import GenerationConfig
from .huggingface import AutoCausalLM

TokenSequence = Union[List[int], torch.LongTensor, torch.Tensor, BatchEncoding]


class Args:
    def __init__(
        self,
        model,
        feature,
        aggressive_memory=False,
        original_max_position_embeddings=None,
        sliding_window_attention=None,
        small_scale=1.0,
        max_position_embeddings=None,
        cache_dir=None,
        flash_attn=True,
        method="pi",
        s_pi_para="./evolution/dim_mono/result_alpha/dim_mono_8192_result.csv",
        tmps="su",
        factor=None,
        finetuned=False,
        stream=0,
        peft_model=None,
        use_cache=False,
        aggressive_mem_causal_lm=False,
        aggressive_mem_decoder=False,
        aggressive_mem_attn=False
    ):
        self.model = [model]
        self.feature = feature
        self.aggressive_memory = aggressive_memory
        self.original_max_position_embeddings = original_max_position_embeddings
        self.sliding_window_attention = sliding_window_attention
        self.small_scale = small_scale
        self.max_position_embeddings = max_position_embeddings
        self.cache_dir = cache_dir
        self.flash_attn = flash_attn
        self.method = method
        self.s_pi_para = s_pi_para
        self.tmps = tmps
        self.factor = factor
        self.finetuned = finetuned
        self.stream = stream
        self.peft_model = peft_model
        self.use_cache = use_cache
        self.aggressive_mem_causal_lm = aggressive_mem_causal_lm
        self.aggressive_mem_decoder = aggressive_mem_decoder
        self.aggressive_mem_attn = aggressive_mem_attn


class sPiAutoLM(AutoCausalLM):
    AUTO_TOKENIZER_CLASS = transformers.AutoTokenizer

    def __init__(
        self,
        model,
        batch_size=1,
        feature=None,
        aggressive_memory=False,
        original_max_position_embeddings=None,
        sliding_window_attention=None,
        small_scale=1.0,
        max_position_embeddings=None,
        cache_dir=None,
        flash_attn=True,
        method="pi",
        s_pi_para="./evolution/dim_mono/result_alpha/dim_mono_8192_result.csv",
        tmps="su",
        factor=None,
        finetuned=False,
        stream=0,
        peft_model=None,
        use_cache=False,
        device="cuda:0"
    ):
        super().__init__(pretrained=model, batch_size=batch_size)
        
        args = Args(
            model, feature, aggressive_memory, original_max_position_embeddings, sliding_window_attention,
            small_scale, max_position_embeddings, cache_dir, flash_attn, method, s_pi_para, tmps, factor,
            finetuned, stream, peft_model, use_cache
        )

        self.tokenizer = self.AUTO_TOKENIZER_CLASS.from_pretrained(
            model, model_max_length=sys.maxsize, trust_remote_code=True, use_fast = False)
        
        self.tokenizer.pad_token = self.tokenizer.eos_token

        torch.cuda.empty_cache()

        if "Mistral" in model or "mistral" in model:
            print(model)
            sys.path.append("/app/s-PI")
            from evaluation.model_loader_mistral import load_model_and_apply_patches_mistral
            self.model = load_model_and_apply_patches_mistral(model, args)
        else:
            print(model)
            sys.path.append("/app/s-PI")
            from evaluation.model_loader_2 import load_model_and_apply_patches
            self.model = load_model_and_apply_patches(model, args)
        
        self.model.eval()
        torch.set_grad_enabled(False)

        self._device = device
        try:
            self.model.to(self._device)
        except:
            print("Failed to place model onto specified device. This may be because the model is quantized via `bitsandbytes`. If the desired GPU is being used, this message is safe to ignore.")
