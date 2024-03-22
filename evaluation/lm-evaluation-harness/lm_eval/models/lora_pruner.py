import os
import math
import torch
import torch.nn.functional as F
import transformers
from pathlib import Path
from typing import List, Mapping, NewType, Optional, Tuple, Union
from tqdm import tqdm

from transformers import BatchEncoding
from transformers import GenerationConfig

from lm_eval import utils
from lm_eval.base import BaseLM

TokenSequence = Union[List[int], torch.LongTensor, torch.Tensor, BatchEncoding]

_DeviceMapping = NewType("DeviceMapping", Mapping[str, Union[int, str, torch.device]])

def get_mask_dict(model):
    mask_dict = {}
    for n,v in model.named_parameters():
        if len(n.split('.')) > 3 and (n.split('.')[3] == 'mlp' or n.split('.')[3] == 'self_attn'):
            print(n)
            mask = (abs(v) > 0)
            mask_dict[n] = mask.cuda()
    return mask_dict

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

PROMPT_DICT_LENGTH = {
    "long": 256,
    "eval_long": 256,
    "middle": 168,
    "short": 130,
}

  
def set_lora_args(config, lora_param):
    config.use_lora = True
    config.lora_rank = 8
    config.lora_train_bias = None
    config.lora_alpha = 8.0
    config.lora_param = lora_param
    config.lora_layers = config.num_hidden_layers
    return config

def _get_dtype(
    dtype: Union[str, torch.dtype], config: Optional[transformers.AutoConfig] = None
) -> torch.dtype:
    """Converts `dtype` from `str` to torch.dtype when possible."""
    if dtype is None and config is not None:
        _torch_dtype = config.torch_dtype
    elif isinstance(dtype, str) and dtype != "auto":
        # Convert `str` args torch dtype: `float16` -> `torch.float16`
        _torch_dtype = getattr(torch, dtype)
    else:
        _torch_dtype = dtype
    return _torch_dtype


class HuggingFaceAutoLM(BaseLM):
    import sys
    # sys.path.append("/home/jiahangxu/working/LoRaPruner")
    from models.modeling_llama import LlamaConfig
    from models.tokenization_llama import LlamaTokenizer
    from models.modeling_llama import LlamaForCausalLM
    AUTO_CONFIG_CLASS: transformers.AutoConfig = LlamaConfig
    AUTO_TOKENIZER_CLASS: transformers.AutoTokenizer = LlamaTokenizer
    AUTO_MODEL_CLASS: transformers.AutoModel = LlamaForCausalLM
    AUTO_PEFT_CLASS = LlamaForCausalLM

    # Default max sequence length setting for when no `max_length` is provided
    # or no max length config setting is found in the model or tokenizer.
    _DEFAULT_MAX_LENGTH: int = 2048

    def __init__(
        self,
        pretrained: str,
        quantized: Optional[Union[bool, str]] = False,
        tokenizer: Optional[str] = None,
        subfolder: Optional[str] = None,
        revision: Optional[str] = "main",
        batch_size: Optional[Union[int, str]] = 1,
        max_batch_size: Optional[int] = 512,
        max_gen_toks: Optional[int] = 256,
        max_length: Optional[int] = None,
        add_special_tokens: Optional[bool] = False,
        use_accelerate: Optional[bool] = False,
        low_cpu_mem_usage: Optional[bool] = True,
        device_map_option: Optional[str] = "auto",
        max_memory_per_gpu: Optional[Union[int, str]] = None,
        max_cpu_memory: Optional[Union[int, str]] = None,
        offload_folder: Optional[str] = "./offload",
        dtype: Optional[Union[str, torch.dtype]] = None,
        device: Optional[Union[int, str]] = "cuda",
        peft: str = None,
        load_in_8bit: Optional[bool] = False,
        load_in_4bit: Optional[bool] = False,
        trust_remote_code: Optional[bool] = False,
        gptq_use_triton: Optional[bool] = False,
        bnb_4bit_quant_type: Optional[str] = None,
        bnb_4bit_compute_dtype: Optional[Union[str, torch.dtype]] = None,
        bnb_4bit_use_double_quant: Optional[bool] = False,
        prompt_mark: str = "0",
        lora_merged: Optional[bool] = False,
        peft_mode: Optional[bool] = False,
        lora_param: str = "Q.V",
    ):
        """Initializes a HuggingFace `AutoModel` and `AutoTokenizer` for evaluation.
        Args:
            pretrained (str):
                The HuggingFace Hub model ID name or the path to a pre-trained
                model to load. This is effectively the `pretrained_model_name_or_path`
                argument of `from_pretrained` in the HuggingFace `transformers` API.
            quantized (str or bool, optional, defaults to False):
                File name of a GPTQ quantized model to load. Set to `True` to use the
                default name of the quantized model.
            add_special_tokens (bool, optional, defaults to True):
                Whether to add special tokens to the input sequences. If `None`, the
                default value will be set to `True` for seq2seq models (e.g. T5) and
                `False` for causal models.
                WARNING: Evaluating causal models with `add_special_tokens=True` is
                currently __not__ supported.
            > Large model loading `accelerate` arguments
            use_accelerate (bool, optional, defaults to False):
                If True, uses the `accelerate` library to load a large model across
                multiple devices.
            low_cpu_mem_usage (bool, optional, defaults to True):
                It True, uses the `accelerate` library to accelerate loading the model.
            device_map_option (str, optional, defaults to "auto"):
                The device map option to use when loading the model with
                `accelerate`.
                Options:
                    "auto", "balanced", "balanced_low_0", "sequential"
                See the `accelerate` docs for more details on these options:
                https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained.device_map
            max_memory_per_gpu (Union[int, str], optional, defaults to None):
                The maximum memory available for each GPU in bytes as `int` or in
                the format f"{significand}{unit_symbol}" where {unit_symbol} is
                any of ["GB", "MB", "GIB", "MIB"]. Refer to the `max_memory` arg in
                the "Parameters for big model inference" section of the following
                docs:
                https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained.max_memory
            max_cpu_memory (Union[int, str], optional, defaults to None):
                The maximum available CPU RAM in bytes as `int` or in the format
                f"{significand}{unit_symbol}" where {unit_symbol} is any of
                ["GB", "MB", "GIB", "MIB"]. Refer to the `max_memory` arg in the
                "Parameters for big model inference" section of the following docs:
                https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained.max_memory
            offload_folder (str, optional, defaults to "./offload"):
                The folder to offload weights into if `device_map` contains any
                "disk" value.
            dtype (Union[str, torch.dtype], optional, defaults to None):):
                Converts the model weights to `dtype`, if specified. Strings get
                converted to `torch.dtype` objects (e.g. `float16` -> `torch.float16`).
                Use `dtype="auto"` to derive the type from the model’s weights.
            peft (str, optional, defaults to None):
                Path of the adapter weights to load from Huggingface. This will usually
                include a directory that includes the files `adapter_config.json` and
                `adapter_model.bin`. Compatible with [PEFT](https://github.com/huggingface/peft)
            load_in_8bit (bool, optional, defaults to False):
                If True, will convert the loaded model into mixed-8bit quantized model. See:
                https://huggingface.co/docs/transformers/main/en/main_classes/quantization#load-a-large-model-in-8bit
            load_in_4bit (bool, optional, defaults to False):
                If True, will convert the loaded model into mixed-4bit quantized model. See:
                https://huggingface.co/docs/transformers/main/en/main_classes/quantization#load-a-large-model-in-4bit
            trust_remote_code (bool, optional, defaults to False):
                If True, will trust the remote code when loading the model.
            gptq_use_triton (bool, optional, defaults to False):
                Use Triton for GPTQ inference.
            bnb_4bit_quant_type (str, optional, defaults to None):
                The quantization type to use for BnB 4bit quantization. See:
                https://github.com/huggingface/transformers/blob/main/src/transformers/utils/quantization_config.py#L77
            bnb_4bit_compute_dtype (Union[str, torch.dtype], optional, defaults to None):
                The compute dtype to use for BnB 4bit quantization. See:
                https://github.com/huggingface/transformers/blob/main/src/transformers/utils/quantization_config.py#L74
            bnb_4bit_use_double_quant (bool, optional, defaults to False):
                Whether or not to use double quant to quantize the absmax.
                https://github.com/huggingface/transformers/blob/main/src/transformers/utils/quantization_config.py#L80

        """
        super().__init__()

        assert isinstance(pretrained, str)
        assert isinstance(device, str)
        assert isinstance(batch_size, (int, str))

        # setup for automatic batch size detection
        if str(batch_size).startswith("auto"):
            batch_size = batch_size.split(":")
            self._batch_size = batch_size[0]
            self.batch_schedule = float(batch_size[1]) if len(batch_size) > 1 else 1
        else:
            self._batch_size = int(batch_size)
        self.max_batch_size = max_batch_size

        self._max_gen_toks = max_gen_toks
        self._max_length = max_length
        self._config = self.AUTO_CONFIG_CLASS.from_pretrained(
            pretrained,
            use_auth_token="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",
            # trust_remote_code=trust_remote_code,
            # revision=revision + ("/" + subfolder if subfolder is not None else ""),
        )
        self._config.use_cache = False
        lora_ckpt = None
        print("lora_param: ", lora_param)
        self._config = set_lora_args(self._config, lora_param)

        self._add_special_tokens = add_special_tokens
        self.tokenizer = self.AUTO_TOKENIZER_CLASS.from_pretrained(
            pretrained,
            use_auth_token="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",
            padding_side="left",
            truncation_side="left",
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.model_max_length = self.max_length

        self.zs = {}
        if peft is not None:
            # import pdb; pdb.set_trace()
            if not lora_merged and not peft_mode:
                lora_ckpt = os.path.join(peft, 'lora_weights.pt')
            if os.path.isfile(os.path.join(peft, 'zs.pt')):
                zs = torch.load(os.path.join(peft, 'zs.pt'), map_location="cpu")

                if zs["head_z"].shape[0] < self._config.num_hidden_layers:
                    if zs["head_z"].shape[0] == 26:
                        zs["head_z"] = torch.concat([torch.ones(4, 1, 32, 1, 1), zs["head_z"], torch.ones(2, 1, 32, 1, 1)])
                        zs["intermediate_z"] = torch.concat([torch.ones(4, 1, 1, 11008), zs["intermediate_z"], torch.ones(2, 1, 1, 11008)])
                    elif zs["head_z"].shape[0] == 28:
                        zs["head_z"] = torch.concat([torch.ones(3, 1, 32, 1, 1), zs["head_z"], torch.ones(1, 1, 32, 1, 1)])
                        zs["intermediate_z"] = torch.concat([torch.ones(3, 1, 1, 11008), zs["intermediate_z"], torch.ones(1, 1, 1, 11008)])

                if "layer_z" in zs:
                    zs['head_layer_z'] = zs['layer_z']
                    zs['mlp_z'] = zs['layer_z']
                    zs.pop('layer_z')
                for key in zs:
                    zs[key] = zs[key].cuda().detach().half()
                self.zs = zs
                
                
        # import pdb; pdb.set_trace()
        self.model = self.AUTO_MODEL_CLASS.from_pretrained(
                self.AUTO_MODEL_CLASS,
                pretrained,
                from_tf=False,
                config=self._config,
                use_auth_token="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",
                lora_ckpt = lora_ckpt
            )
        mask_dict = get_mask_dict(self.model)
        self.zs["mask_dict"] = mask_dict
        if peft_mode:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(
                self.model,
                peft,
                torch_dtype=torch.float16,
            )
            
        
        self.model.half()
        self.model.eval()
        torch.set_grad_enabled(False)

        self._device = device
        try:
            self.model.to(self._device)
        except:
            print("Failed to place model onto specified device. This may be because the model is quantized via `bitsandbytes`. If the desired GPU is being used, this message is safe to ignore.")

        if prompt_mark == "1" or prompt_mark == 1:
            prompt_mark = "long"
        elif prompt_mark == "1-1":
            prompt_mark = "eval_long"
        elif prompt_mark == "2" or prompt_mark == 2:
            prompt_mark = "middle"
        elif prompt_mark == "3" or prompt_mark == 3:
            prompt_mark = "short"
        else:
            prompt_mark = None
        print("prompt_mark: ", prompt_mark)
        
        if prompt_mark in ["long", "middle", "short", "eval_long"]:
            prompt = self.tokenizer(PROMPT_DICT[f"prompt_{prompt_mark}_pruning"])
        else:
            prompt = {'input_ids': [], 'attention_mask': []}
        self.prompt_input_ids = torch.tensor(prompt["input_ids"]).to(self._device)

    @property
    def add_special_tokens(self) -> bool:
        """Whether to include special tokens in encoded text. This should be
        determined by whether or not the model was trained with special tokens.
        TODO: Remove these conditionals once HuggingFace supports a way to
        check whether or not an arbitrary model was trained with special tokens.
        """
        if self._add_special_tokens is not None:
            return self._add_special_tokens
        elif self.AUTO_MODEL_CLASS is transformers.AutoModelForCausalLM:
            return False
        elif self.AUTO_MODEL_CLASS is transformers.AutoModelForSeq2SeqLM:
            return True
        else:
            raise ValueError(
                "Could not determine `add_special_tokens` value from the model "
                "class. Set to `True` or `False` depending on whether the model "
                "was pre-trained with special tokens."
            )

    @property
    def eot_token(self) -> str:
        return self.tokenizer.eos_token

    @property
    def eot_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def max_gen_toks(self) -> int:
        return self._max_gen_toks

    @property
    def max_length(self) -> int:
        """Return the maximum sequence length of the model.
        NOTE: Different model configurations have different max sequence length
        attribute names.
            - n_positions: (CTRLConfig, T5Config)
            - max_position_embeddings: (BartConfig, RoFormerConfig)
            - n_ctx: (GPT2Config)
        NOTE: For relative position encoded models you should specify the max
        sequence length of the model in the constructor via `max_length`.
        """
        if self._max_length is not None:
            return self._max_length
        # Try to get the sequence length from the model config.
        seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
        for attr in seqlen_config_attrs:
            if hasattr(self._config, attr):
                return getattr(self._config, attr)
        if hasattr(self.tokenizer, "model_max_length"):
            if self.tokenizer.model_max_length == 1000000000000000019884624838656:
                return self._DEFAULT_MAX_LENGTH
            return self.tokenizer.model_max_length
        return self._DEFAULT_MAX_LENGTH

    @property
    def batch_size(self) -> int:
        # TODO: Add adaptive batch size.
        return self._batch_size  # * gpus

    @property
    def device(self) -> Union[int, str, torch.device]:
        return self._device

    def tok_encode(self, string: str) -> TokenSequence:
        # TODO: Merge `tok_encode_batch` here.
        return self.tokenizer.encode(string, add_special_tokens=self.add_special_tokens)

    def tok_encode_batch(self, strings: List[str]) -> TokenSequence:
        return self.tokenizer(
            strings,
            padding=True,
            add_special_tokens=self.add_special_tokens,
            return_tensors="pt",
        )

    def tok_decode(self, tokens: torch.LongTensor) -> List[str]:
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)

    def greedy_until(self, requests):
        # TODO: implement fully general `until` that handles until that are
        #       multiple tokens or that span multiple tokens correctly

        # TODO: extract to TokenizedLM?
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end

            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        re_ord = utils.Reorderer(requests, _collate)

        warn_stop_seq = False
        for context, request_args in tqdm(re_ord.get_reordered()):
            until = request_args["until"]
            if isinstance(until, str):
                until = [until]

            if until:
                try:
                    (primary_until,) = self.tok_encode(until[0])
                except ValueError:
                    if not warn_stop_seq:
                        print(
                            "Warning: a primary stop sequence is multi-token! Will default to EOS token for this tokenizer. Consider using `hf-causal-experimental` for multi-token stop sequence support for the time being."
                        )
                        warn_stop_seq = True
                    primary_until = self.eot_token_id
            else:
                primary_until = None

            context_enc = torch.tensor(
                [self.tok_encode(context)[self.max_gen_toks - self.max_length :]]
            ).to(self.device)

            max_gen_tokens = min(
                self.max_gen_toks, request_args.get("max_length", self.max_gen_toks)
            )
            cont = self._model_generate(
                context_enc, context_enc.shape[1] + max_gen_tokens, primary_until
            )

            s = self.tokenizer.decode(cont[0].tolist()[context_enc.shape[1] :])

            for term in until:
                s = s.split(term)[0]

            # partial caching
            self.cache_hook.add_partial("greedy_until", (context, until), s)

            res.append(s)

        return re_ord.get_original(res)


class GenerationConfig_my(GenerationConfig): 
    def validate(self):
        if self.early_stopping not in {True, False, "never"}:
            raise ValueError(f"`early_stopping` must be a boolean or 'never', but is {self.early_stopping}.")

    def update(self, **kwargs):
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                to_remove.append(key)

        # remove all the attributes that were updated, without modifying the input dict
        unused_kwargs = {key: value for key, value in kwargs.items() if key not in to_remove}
        return unused_kwargs

class AutoCausalLM(HuggingFaceAutoLM):
    """Causal language modeling.
    You can find a set of supported models in the HF documentation:
    https://huggingface.co/docs/transformers/main/model_doc/auto#transformers.AutoModelForCausalLM
    """

    def _model_call(
        self, inputs: TokenSequence, labels: Optional[TokenSequence] = None
    ) -> TokenSequence:
        inputs = torch.concat([self.prompt_input_ids.unsqueeze(0).to(inputs.dtype), inputs], dim=-1)
        return self.model(inputs, **self.zs)["logits"]

    def _model_generate(self, context, max_length, eos_token_id):
        context = torch.concat([self.prompt_input_ids.unsqueeze(0).to(context.dtype), context], dim=-1)
        generation_kwargs = {"do_sample": False, "max_length": max_length, "max_new_tokens": max_length - context.shape[-1]}
        #TODO: generation_config
        generation_config = GenerationConfig_my()
        
        if eos_token_id is not None:
            generation_kwargs['eos_token_id'] = eos_token_id
            generation_kwargs['pad_token_id'] = eos_token_id # setting eos_token_id as pad token
        generation_config.update(**generation_kwargs)
        return self.model.generate(
            context,
            zs=self.zs,
            generation_config=generation_config,
        ).sequences


class AutoSeq2SeqLM(HuggingFaceAutoLM):
    """Seq2Seq language modeling.
    You can find a set of supported models in the following documentation:
    https://huggingface.co/docs/transformers/main/model_doc/auto#transformers.AutoModelForSeq2SeqLM
    """

    def loglikelihood(
        self, requests: List[Tuple[str, str]]
    ) -> List[Tuple[float, bool]]:
        raise NotImplementedError
        new_requests = []
        for chunk in utils.chunks(requests, self.batch_size):
            context, continuation = zip(*chunk)

            # Fill empty contexts with the EOT token.
            context = [
                f"{self.eot_token}" if len(text) == 0 else text for text in context
            ]
            context_enc = self.tok_encode_batch(context)
            for key in context_enc:
                context_enc[key] = context_enc[key][:, -self.max_length :]

            # Remove leading whitespace introduced by the default
            # `text_target_separator` since the context and continuation
            # will not be concatenated as a single (decoder) input.
            continuation = [text.lstrip() for text in continuation]
            continuation_enc = self.tok_encode_batch(list(continuation))
            for key in continuation_enc:
                continuation_enc[key] = continuation_enc[key][:, -self.max_length :]

            new_requests.append(
                ((context, continuation), context_enc, continuation_enc)
            )
        return self._loglikelihood_tokens(new_requests)

    def loglikelihood_rolling(self, requests: List[Tuple[str, str]]) -> List[float]:
        raise NotImplementedError
        loglikelihoods = []
        for (string,) in tqdm(requests):
            rolling_token_windows = list(
                map(
                    utils.make_disjoint_window,
                    utils.get_rolling_token_windows(
                        token_list=self.tok_encode(string),
                        prefix_token=self.eot_token_id,
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                )
            )
            contexts, conts = utils.split_and_pad_windows(
                rolling_token_windows,
                pad_token_id=self.eot_token_id,
                max_seq_len=self.max_length,
            )
            # Manually create BatchEncoding tensors with attention masks as
            # expected by `self._model_call` in `self._loglikelihood_tokens`.
            contexts_enc = torch.Tensor(contexts).long()
            contexts_enc = transformers.tokenization_utils_base.BatchEncoding(
                {
                    "input_ids": contexts_enc,
                    "attention_mask": (contexts_enc != self.eot_token_id).long(),
                }
            )
            conts_enc = torch.Tensor(conts).long()
            conts_enc = transformers.tokenization_utils_base.BatchEncoding(
                {
                    "input_ids": conts_enc,
                    "attention_mask": (conts_enc != self.eot_token_id).long(),
                }
            )
            # TODO: Extract out this call so it only gets called once and also
            # somehow figure out partial caching for.
            rolling_token_windows_request = [
                ((contexts, conts), contexts_enc, conts_enc)
            ]
            string_nll = self._loglikelihood_tokens(
                rolling_token_windows_request, disable_tqdm=True
            )
            string_nll = [x[0] for x in string_nll]  # discard is_greedy
            string_nll = sum(string_nll)
            loglikelihoods.append(string_nll)
        return loglikelihoods

    def _loglikelihood_tokens(
        self,
        requests: List[Tuple[Tuple[str, str], TokenSequence, TokenSequence]],
        disable_tqdm: Optional[bool] = False,
    ) -> List[Tuple[float, bool]]:
        raise NotImplementedError
        results = []
        
        for chunk in tqdm(
            requests, total=math.ceil(len(requests)), disable=disable_tqdm
        ):
            cache_keys, inputs_tokens, targets_tokens = chunk
            inputs_tokens = inputs_tokens.to(self.device)
            targets_tokens = targets_tokens.to(self.device)
            outputs = self._model_call(inputs=inputs_tokens, labels=targets_tokens)
            log_softmaxes = F.log_softmax(outputs.logits, dim=-1)

            output_iterator = zip(
                zip(cache_keys[0], cache_keys[1]),
                log_softmaxes,
                targets_tokens["input_ids"],
                targets_tokens["attention_mask"],
            )
            for cache_key, log_softmax, target_tokens, target_mask in output_iterator:
                length = target_mask.sum()
                log_softmax = log_softmax[:length]
                target_tokens = target_tokens[:length]
                greedy_tokens = log_softmax.argmax(dim=-1)
                max_equal = (greedy_tokens == target_tokens).all()
                target_logits = torch.gather(
                    log_softmax, 1, target_tokens.unsqueeze(-1)
                ).squeeze(-1)
                answer = (float(target_logits.sum()), bool(max_equal))
                results.append(answer)
                if cache_key is not None:
                    self.cache_hook.add_partial("loglikelihood", cache_key, answer)
        return results

    def _model_call(
        self, inputs: TokenSequence, labels: Optional[TokenSequence] = None
    ) -> TokenSequence:
        raise NotImplementedError
        return self.model(**inputs, labels=labels["input_ids"])

    def _model_generate(
        self,
        inputs: transformers.BatchEncoding,
        max_tokens: int,
        stop: Optional[List[str]] = None,
    ) -> TokenSequence:
        raise NotImplementedError
        input_ids = inputs["input_ids"][:, -self.max_length :].to(self.device)
        attention_mask = inputs["attention_mask"][:, -self.max_length :].to(self.device)

        # Generate one token to calculate the number of start tokens prepended to decoder_input_ids
        # (leaving this here in case the below assumption is violated in the future)
        # one_tok_gen = self.model.generate(
        #    input_ids=torch.zeros((1, 1), dtype=torch.int),
        #    min_length=2,
        #    max_new_tokens=1,
        # ).squeeze()
        # initial_decoder_input_length = len(one_tok_gen) - 1

        # Assume that there will always only be one token in the decoder inputs, assumption holds for existing HF models
        stopping_criteria = stop_sequences_criteria(
            self.tokenizer, stop, 1, input_ids.shape[0]
        )

        generations = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            stopping_criteria=stopping_criteria,
            do_sample=False,
        )
        return generations


class MultiTokenEOSCriteria(transformers.StoppingCriteria):
    """Criteria to stop on the specified multi-token sequence."""

    def __init__(
        self,
        sequence: str,
        tokenizer: transformers.PreTrainedTokenizer,
        initial_decoder_input_length: int,
        batch_size: int,
    ):
        self.initial_decoder_input_length = initial_decoder_input_length
        self.done_tracker = [False] * batch_size
        self.sequence = sequence
        self.sequence_ids = tokenizer.encode(sequence, add_special_tokens=False)
        self.sequence_id_len = len(self.sequence_ids)
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        # For efficiency, we compare the last n tokens where n is the number of tokens in the stop_sequence
        lookback_ids_batch = input_ids[:, self.initial_decoder_input_length :][
            :, -self.sequence_id_len :
        ]

        lookback_tokens_batch = self.tokenizer.batch_decode(lookback_ids_batch)

        for i, done in enumerate(self.done_tracker):
            if not done:
                self.done_tracker[i] = self.sequence in lookback_tokens_batch[i]
        return False not in self.done_tracker


def stop_sequences_criteria(
    tokenizer: transformers.PreTrainedTokenizer,
    stop_sequences: List[str],
    initial_decoder_input_length: int,
    batch_size: int,
) -> transformers.StoppingCriteriaList:
    return transformers.StoppingCriteriaList(
        [
            *[
                MultiTokenEOSCriteria(
                    sequence, tokenizer, initial_decoder_input_length, batch_size
                )
                for sequence in stop_sequences
            ],
        ]
    )
