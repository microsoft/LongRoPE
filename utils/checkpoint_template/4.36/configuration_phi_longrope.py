from transformers.utils import logging
# LLAMA COMPONENTS BEGIN
from transformers.models.llama.configuration_llama import LlamaConfig as BaseConfig
# LLAMA COMPONENTS END
# MISTRAL COMPONENTS BEGIN
from transformers.models.mistral.configuration_mistral import MistralConfig as BaseConfig
# MISTRAL COMPONENTS END


logger = logging.get_logger(__name__)


class PhiLongRoPEConfig(BaseConfig):

    model_type = "phi_longrope"

    def __init__(
        self,
        original_max_position_embeddings=4096,
        **kwargs,
    ):
        self.original_max_position_embeddings = original_max_position_embeddings
        super().__init__(**kwargs)

    def _rope_scaling_validation(self):
        if self.rope_scaling is None:
            return

        assert (isinstance(self.rope_scaling, dict)) and \
            ("type" in self.rope_scaling) and \
            ("short_factor" in self.rope_scaling) and \
            ("long_factor" in self.rope_scaling), \
            f"`rope_scaling` must be a dictionary with with three fields, `type`, `short_factor` and `long_factor`, " \
            f"got {self.rope_scaling}"

        assert self.rope_scaling["type"].lower() == "longrope", \
            f"RoPE scaling type must be `longrope`"

        short_factor = self.rope_scaling["short_factor"]
        assert isinstance(short_factor, list) and \
            all([isinstance(x, (int, float)) for x in short_factor]), \
            f"RoPE scaling factor must be a list of numbers, got {short_factor}"
        assert len(short_factor) == self.hidden_size // self.num_attention_heads // 2, \
            f"the length of RoPE scaling factor must be half of the attention head, got {short_factor}"

        long_factor = self.rope_scaling["long_factor"]
        assert isinstance(long_factor, list) and \
            all([isinstance(x, (int, float)) for x in long_factor]), \
            f"RoPE scaling factor must be a list of numbers, got {long_factor}"
        assert len(long_factor) == self.hidden_size // self.num_attention_heads // 2, \
            f"the length of RoPE scaling factor must be half of the attention head, got {long_factor}"
