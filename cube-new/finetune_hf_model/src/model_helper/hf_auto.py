from fairseq.models import register_model
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForQuestionAnswering,
)

from .common import HFLMConfig
from .hf_base import BaseFairseqWrapper, build_wrapped_model


@register_model('hf_auto_lm', HFLMConfig)
class AutoFairseqWrapper(BaseFairseqWrapper):
    @classmethod
    def build_model(cls, args, task):
        assert args.llm_model_name_or_path
        return build_wrapped_model(args, AutoModel, cls)


@register_model('hf_auto_causal_lm', HFLMConfig)
class CausalLMFairseqWrapper(BaseFairseqWrapper):
    @classmethod
    def build_model(cls, args, task):
        assert args.llm_model_name_or_path
        return build_wrapped_model(args, AutoModelForCausalLM, cls)


@register_model('hf_auto_seq2seq_lm', HFLMConfig)
class Seq2SeqFairseqWrapper(BaseFairseqWrapper):
    @classmethod
    def build_model(cls, args, task):
        assert args.llm_model_name_or_path
        return build_wrapped_model(args, AutoModelForSeq2SeqLM, cls)


@register_model('hf_auto_seq_cls_lm', HFLMConfig)
class SeqClsFairseqWrapper(BaseFairseqWrapper):
    @classmethod
    def build_model(cls, args, task):
        assert args.llm_model_name_or_path
        return build_wrapped_model(args, AutoModelForSequenceClassification, cls)


@register_model('hf_auto_token_cls_lm', HFLMConfig)
class TokenClsFairseqWrapper(BaseFairseqWrapper):
    @classmethod
    def build_model(cls, args, task):
        assert args.llm_model_name_or_path
        return build_wrapped_model(args, AutoModelForTokenClassification, cls)


@register_model('hf_auto_qa_lm', HFLMConfig)
class QAFairseqWrapper(BaseFairseqWrapper):
    @classmethod
    def build_model(cls, args, task):
        assert args.llm_model_name_or_path
        return build_wrapped_model(args, AutoModelForQuestionAnswering, cls)
