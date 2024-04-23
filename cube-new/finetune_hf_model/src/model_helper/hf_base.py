import logging
from typing import Optional, Type

import torch

from transformers import PreTrainedModel
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
)
from fairseq.models import BaseFairseqModel

from ..data_helper import get_tokenizer
from .common import (
    TASK_TYPE,
    wrap_model_with_lora
)

_logger = logging.getLogger(__name__)


class BaseFairseqWrapper(BaseFairseqModel):
    def __init__(self, model: PreTrainedModel, pad_ids: int) -> None:
        super().__init__()
        assert isinstance(pad_ids, int)
        self.hf_model = model
        self.pad_ids = pad_ids
        self.hf_model.config.return_dict = False

    def forward(self, src_tokens: torch.Tensor, **kwargs):
        # Based on experience, almost all huggingface language models have attention_mask parameter.
        # If you find exceptions, please modify the logic here.
        if 'attention_mask' not in kwargs:
            kwargs['attention_mask'] = (src_tokens != self.pad_ids)
        res = self.hf_model(src_tokens, **kwargs)
        return res[0]


def auto_detect_model_type(model: PreTrainedModel) -> Optional[TASK_TYPE]:
    task_types = ('SEQ_CLS', 'SEQ_2_SEQ_LM', 'CAUSAL_LM', 'TOKEN_CLS', 'QUESTION_ANS')
    model_mappings = (
        MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
        MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
        MODEL_FOR_CAUSAL_LM_MAPPING,
        MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
        MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    )
    for task_type, model_mapping in zip(task_types, model_mappings):
        if type(model.config) in model_mapping:
            if isinstance(model, model_mapping[type(model.config)]):
                return task_type
    return None


def build_wrapped_model(args, model_cls: Type[PreTrainedModel], wrapper_cls: Type[BaseFairseqWrapper], task_type: Optional[TASK_TYPE] = None) -> BaseFairseqWrapper:
    # load model & tokenizer
    model: PreTrainedModel = model_cls.from_pretrained(args.llm_model_name_or_path, torch_dtype='auto', device_map='cpu')
    tokenizer = get_tokenizer(args.llm_model_name_or_path)

    if task_type is None:
        task_type = auto_detect_model_type(model)

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

    # if using flash attention with optimum
    if args.use_fast_kernels:
        try:
            from optimum.bettertransformer import BetterTransformer
            model = BetterTransformer.transform(model)
        except ImportError:
            _logger.warning("Module 'optimum' not found. Please install 'optimum' it before proceeding.")

    # if using lora with peft
    if args.use_lora:
        if task_type is None:
            raise RuntimeError(f'can not detect the task type of model with type {type(model)}')
        model = wrap_model_with_lora(args, model, task_type)

    return wrapper_cls(model, pad_ids=tokenizer.pad_token_id)
