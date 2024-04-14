from fairseq.models import register_model

from ..common import HFLMConfig
from ..hf_base import BaseFairseqWrapper, build_wrapped_model

import logging

logger = logging.getLogger(__name__)


try:
    from .modeling_cube_llama import CubeLlamaForCausalLM

    @register_model('hf_cube_llama', HFLMConfig)
    class CubeLlamaWrapper(BaseFairseqWrapper):
        @classmethod
        def build_model(cls, args, task):
            assert args.llm_model_name_or_path
            return build_wrapped_model(args, CubeLlamaForCausalLM, cls)
except:
    logger.info('Can not register hf_cube_llama, it is based on transformers==4.38.1')
finally:
    pass
