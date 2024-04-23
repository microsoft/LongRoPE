from dataclasses import dataclass, field
import os
import torch
from transformers import AutoConfig

from fairseq.dataclass import FairseqDataclass
from fairseq.models import register_model

from ..hf_base import BaseFairseqWrapper
from ...data_helper import get_tokenizer

import logging
from ...rope import load_model

logger = logging.getLogger(__name__)


from .modeling_cube_llama import CubeLlamaForCausalLM
from .modeling_cube_mistral import CubeMistralForCausalLM

@dataclass
class GenAIConfig(FairseqDataclass):
    llm_model_name_or_path: str = field(default='', metadata={'help': 'Huggingface model id or path, i.e., meta-llama/Llama-2-7b-hf or /home/USER/llama2'})
    use_fast_kernels: bool = field(default=False)
    fp16: bool = field(default=False)
    bf16: bool = field(default=False)


@register_model('hf_cube_genai', GenAIConfig)
class CubeGenAIWrapper(BaseFairseqWrapper):
    @classmethod
    def build_model(cls, args, task):
        assert args.llm_model_name_or_path
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

        model = load_model(
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

        tokenizer = get_tokenizer(args.llm_model_name_or_path,
                                    default_bos_token="<s>",
                                    default_eos_token="</s>",
                                    default_pad_token="[PAD]",
                                    default_unk_token="<unk>")

        # smart resize the embedding dim
        if len(tokenizer) - model.vocab_size > 0:
            model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=256)

            input_embeddings = model.get_input_embeddings().weight.data
            old_input_embed_len = input_embeddings.shape[0]
            input_embeddings_avg = input_embeddings[:old_input_embed_len].mean(dim=0, keepdim=True)
            input_embeddings[old_input_embed_len:] = input_embeddings_avg

            if model.get_output_embeddings() is not None:
                output_embeddings = model.get_output_embeddings().weight.data
                old_output_embed_len = output_embeddings.shape[0]        
                output_embeddings_avg = output_embeddings[:old_output_embed_len].mean(dim=0, keepdim=True)
                output_embeddings[old_output_embed_len:] = output_embeddings_avg

        return CubeGenAIWrapper(model, tokenizer.pad_token_id)

