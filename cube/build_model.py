import os
import torch
from dataclasses import dataclass, field

from transformers import PreTrainedModel, AutoConfig

from fairseq.dataclass import FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model

from rope import load_model

import logging
logger = logging.getLogger(__name__)

from build_dict import get_tokenizer
from modeling_cube_llama import CubeLlamaForCausalLM
from modeling_cube_mistral import CubeMistralForCausalLM


@dataclass
class HFCausalLMConfig(FairseqDataclass):
    llm_model_name_or_path: str = field(default='', metadata={'help': 'Huggingface model id or path.'})
    use_fast_kernels: bool = field(default=False)
    fp16: bool = field(default=False)
    bf16: bool = field(default=False)


@register_model('hf_causal_lm', HFCausalLMConfig)
class CausalLMFairseqWrapper(BaseFairseqModel):
    def __init__(self, model: PreTrainedModel, pad_ids: int) -> None:
        super().__init__()
        assert isinstance(pad_ids, int)
        self.hf_model = model
        self.pad_ids = pad_ids
        self.hf_model.config.return_dict = False

    def forward(self, src_tokens: torch.Tensor, **kwargs):
        attention_mask = (src_tokens != self.pad_ids)
        res = self.hf_model.forward(input_ids=src_tokens, attention_mask=attention_mask)
        return res[0]

    @classmethod
    def build_model(cls, args, task):
        assert args.llm_model_name_or_path
        return build_wrapped_causal_lm_model(args)


def build_wrapped_causal_lm_model(args) -> CausalLMFairseqWrapper:
    model_classes = {
        'llama': CubeLlamaForCausalLM,
        'mistral': CubeMistralForCausalLM,
    }

    config = AutoConfig.from_pretrained(args.llm_model_name_or_path, trust_remote_code=True)

    if args.fp16:
        assert not args.bf16, 'Input --fp16 and --bf16 at the same time'
        config.torch_dtype = "float16"
        assume_dtype = torch.float16
    elif args.bf16:
        assert not args.fp16, 'Input --fp16 and --bf16 at the same time'
        config.torch_dtype = "bfloat16"
        assume_dtype = torch.bfloat16
    else:
        logger.warning('Setting dtype=torch.float32 by default')
        assume_dtype = torch.float32

    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(assume_dtype)

    if hasattr(config, 'use_cache'):
        config.use_cache = False

    if 'ATTN_SLIDING_WINDOW' in os.environ:
        attn_sliding_window = int(os.environ['ATTN_SLIDING_WINDOW'])
    else:
        attn_sliding_window = None

    model: PreTrainedModel = load_model(
        model_name_or_path=args.llm_model_name_or_path,
        rope_method=os.environ['ROPE_METHOD'],
        max_position_embeddings=int(os.environ['SEQ_LEN']),
        model_class=model_classes[config.model_type],
        rope_params={
            'longrope_params_path': os.environ['LONGROPE_PARAMS'],
            'longrope_scaling_policy': os.environ['LONGROPE_SCALING_POLICY'],
        },
        attn_implementation='flash_attention_2',
        attn_sliding_window=attn_sliding_window,
        device_map='cpu',
        config=config,
    )

    torch.set_default_dtype(default_dtype)

    tokenizer = get_tokenizer(args.llm_model_name_or_path, 1024)
    if len(tokenizer) - model.vocab_size > 0:
        model.resize_token_embeddings(len(tokenizer))

    return CausalLMFairseqWrapper(model, pad_ids=tokenizer.pad_token_id)
