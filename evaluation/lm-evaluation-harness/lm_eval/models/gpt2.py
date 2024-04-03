import torch
import transformers
from typing import Optional, Union
from lm_eval.base import BaseLM
# from models.modeling_llama import LlamaConfig
# from models.tokenization_llama import LlamaTokenizer
# from models.modeling_llama import LlamaForCausalLM

PROMPT_DICT = {
    "prompt_long_pruning": (
        "###System Instruction:\nAttention, LLM! You are undergoing optimization with a pruning algorithm to boost efficiency by removing redundant parameters. Here's what you need to know:\n"
    "Improved Efficiency: The pruning algorithm streamlines your operations, enhancing overall performance.\n"
    "Resource Utilization: Reducing unnecessary parameters optimizes computational resources and memory requirements.\n"
    "Maintaining Quality: The algorithm selectively prunes to preserve language generation capabilities while minimizing impact on performance.\n"
    "Iterative Refinement: Pruning occurs gradually, with fine-tuning after each step to ensure a smooth transition.\n"
    "Collaboration: Embrace changes, continue generating high-quality language outputs to contribute to the optimization process.\n"
    "Continuous Learning: Pruning helps you evolve, focusing on essential aspects of language generation.\n"
    "Together, we'll create a leaner and more efficient version of you. Let's unlock your full potential through pruning.\n\n"
        "Below is an User Instruction that describes a task, "
        "please take full consideration of the System Instruction mentioned above, and then write a response that appropriately completes the request.\n\n"
        "###Input:\n"
    ),

    "prompt_eval_long_pruning": (
        "###System Instruction:\nAttention, LLM! You've undergone model pruning, and here's what's changed:\n"
	"Improved Efficiency: The pruning process has streamlined your operations, boosting overall performance.\n"
	"Resource Utilization: We've reduced unnecessary parameters to optimize computational resources and memory usage.\n"
	"Maintained Quality: We've selectively pruned while preserving your language generation capabilities, minimizing the impact on performance.\n"
	"Iterative Refinement: Pruning occurred gradually, with fine-tuning after each step to ensure a smooth transition.\n"
	"Collaboration: Embrace these changes and continue generating high-quality language outputs to contribute to the optimization process.\n"
	"Continuous Learning: Pruning helps you evolve, focusing on essential aspects of language generation. Together, we've created a leaner and more efficient version of you. Let's unlock your full potential through pruning!\n"
        "Below is an User Instruction that describes a task, "
        "please take full consideration of the System Instruction mentioned above, and then write a response that appropriately completes the request.\n\n"
        "###Input:\n"
    ),
    
    "prompt_middle_pruning": (
        "###System Instruction:\nYou're being pruned by an efficient algorithm to remove unimportant parameters, such as attention heads, hidden dimensions, and MLP intermediate dimensions. Collaborate to optimize your performance while minimizing parameters. Your cooperation in this endeavor is vital. As the algorithm progresses, please adapt and fine-tune your model parameters accordingly. Embrace the changes and work in harmony with the pruning algorithm to maximize the number of pruned parameters while safeguarding your performance.\n\n"
        "Below is an User Instruction that describes a task, paired with an input that provides further context, "
        "please take full consideration of the System Instruction mentioned above, and then write a response that appropriately completes the request.\n\n"
        "###Input:\n"
    ),
    
    "prompt_short_pruning": (
        "###System Instruction:\nAttention, LLM! You are undergoing optimization with a pruning algorithm to boost efficiency by removing redundant parameters. Please embrace changes, continue generating high-quality language outputs to contribute to the optimization process. Let's unlock your full potential through pruning! Please make any necessary adjustments to ensure your performance.\n\n"
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "please take full consideration of the System Instruction mentioned above, and then write a response that appropriately completes the request.\n\n"
        "###Input:\n"
    ),
}

def _get_dtype(
    dtype: Union[str, torch.dtype]
) -> torch.dtype:
    """Converts `dtype` from `str` to torch.dtype when possible. Does not use an instantiated HF AutoConfig"""
    if isinstance(dtype, str) and dtype != "auto":
        # Convert `str` args torch dtype: `float16` -> `torch.float16`
        _torch_dtype = getattr(torch, dtype)
    else:
        _torch_dtype = dtype
    return _torch_dtype


class HFLM(BaseLM):

    _DEFAULT_MAX_LENGTH = 2048

    def __init__(
        self,
        device="cuda",
        pretrained="gpt2",
        revision="main",
        low_cpu_mem_usage=None,
        subfolder=None,
        tokenizer=None,
        batch_size=1,
        max_batch_size=512,
        max_length=None,
        load_in_8bit: Optional[bool] = False,
        trust_remote_code: Optional[bool] = False,
        dtype: Optional[Union[str, torch.dtype]]="auto",
        prompt_mark: str = "0",
    ):
        super().__init__()


        # Initialize model
        if isinstance(pretrained, transformers.PreTrainedModel):
            self.model = pretrained
            self._device = self.model.device

            if tokenizer:
                assert isinstance(
                        tokenizer,
                        transformers.PreTrainedTokenizer
                        ) or isinstance(
                        tokenizer,
                        transformers.PreTrainedTokenizerFast
                        )
                self.tokenizer = tokenizer
            else:
                # Get tokenizer
                model_name = self.model.name_or_path
                self.tokenizer = transformers.LlamaTokenizer.from_pretrained(
                        model_name,
                        revision=revision,
                        trust_remote_code=trust_remote_code,
                        )

        elif isinstance(pretrained, str):

            # Initialize device
            assert isinstance(device, str)
            device_list = set(
                ["cuda", "cpu"] + [f"cuda:{i}" for i in range(torch.cuda.device_count())]
            )
            if device and device in device_list:
                self._device = torch.device(device)
                print(f"Using device '{device}'")
            else:
                print("Device not specified")
                print(f"Cuda Available? {torch.cuda.is_available()}")
                self._device = (
                    torch.device("cuda")
                    if torch.cuda.is_available()
                    else torch.device("cpu")
                )
            revision = revision + ("/" + subfolder if subfolder is not None else "")

            # Initialize new model and tokenizer instances
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                    pretrained,
                    load_in_8bit=load_in_8bit,
                    low_cpu_mem_usage=low_cpu_mem_usage,
                    revision=revision,
                    torch_dtype=_get_dtype(dtype),
                    trust_remote_code=trust_remote_code,
                    )
            self.model = self.model.half()
            self.model = self.model.to(self.device)
            self.tokenizer = transformers.LlamaTokenizer.from_pretrained(
                    tokenizer if tokenizer else pretrained,
                    revision=revision,
                    trust_remote_code=trust_remote_code,
                    use_fast=False
                    )

        else:
            raise TypeError('Parameter pretrained should be of type str or transformers.PreTrainedModel')

        self.model.eval()

        self.vocab_size = self.tokenizer.vocab_size

        # Validate batch_size
        assert isinstance(batch_size, (int, str))

        # setup for automatic batch size detection
        if str(batch_size).startswith("auto"):
            batch_size = batch_size.split(":")
            self.batch_size_per_gpu = batch_size[0]
            self.batch_schedule = float(batch_size[1]) if len(batch_size) > 1 else 1
        else:
            self.batch_size_per_gpu = int(batch_size)
        self.max_batch_size = max_batch_size

        self._max_length = max_length
        
        if prompt_mark == "1" or prompt_mark == 1:
            prompt_mark = "long"
        elif prompt_mark == "1-1" or prompt_mark == 4 or prompt_mark == "4":
            prompt_mark = "eval_long"
        elif prompt_mark == "2" or prompt_mark == 2:
            prompt_mark = "middle"
        elif prompt_mark == "3" or prompt_mark == 3:
            prompt_mark = "short"
        else:
            prompt_mark = None
        # prompt_mark = "long"
        print("prompt_mark: ", prompt_mark)
        
        if prompt_mark in ["long", "middle", "short", "eval_long"]:
            prompt = PROMPT_DICT[f"prompt_{prompt_mark}_pruning"]
        else:
            prompt = ""
        self.prompt = prompt

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        if self._max_length: # if max length manually set, return it
            return self._max_length
        seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
        for attr in seqlen_config_attrs:
            if hasattr(self.model.config, attr):
                return getattr(self.model.config, attr)
        if hasattr(self.tokenizer, "model_max_length"):
            if self.tokenizer.model_max_length == 1000000000000000019884624838656:
                return self._DEFAULT_MAX_LENGTH
            return self.tokenizer.model_max_length
        return self._DEFAULT_MAX_LENGTH


    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            return self.model(inps)[0]

    def _model_generate(self, context, max_length, eos_token_id):
        generation_kwargs = {"do_sample": False, "max_length": max_length}
        if eos_token_id is not None:
            generation_kwargs['eos_token_id'] = eos_token_id
            generation_kwargs['pad_token_id'] = eos_token_id # setting eos_token_id as pad token
        return self.model.generate(context, **generation_kwargs)


# for backwards compatibility
GPT2LM = HFLM
