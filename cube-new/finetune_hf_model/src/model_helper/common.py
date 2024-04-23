from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from peft import LoraConfig, get_peft_model, PeftConfig, PeftModel
from fairseq.dataclass import FairseqDataclass

import logging
_logger = logging.getLogger(__name__)


@dataclass
class HFLMConfig(FairseqDataclass):
    llm_model_name_or_path: str = field(default='', metadata={'help': 'Huggingface model id or path, i.e., meta-llama/Llama-2-7b-hf or /home/USER/llama2'})
    # flash attention related, using optimum api
    use_fast_kernels: bool = field(default=False)
    # lora related, using peft api
    use_lora: bool = field(default=False)
    lora_load_path: str = field(default='')
    lora_rank: int = field(default=8)
    lora_alpha: int = field(default=8)
    lora_dropout: float = field(default=0.0)
    lora_fan_in_fan_out: bool = field(default=False)
    lora_bias: str = field(default='none')
    target_modules: str = field(default='')
    modules_to_save: str = field(default='')


TASK_TYPE = Literal['SEQ_CLS', 'SEQ_2_SEQ_LM', 'CAUSAL_LM', 'TOKEN_CLS', 'QUESTION_ANS', 'FEATURE_EXTRACTION']

def wrap_model_with_lora(args, model, task_type: TASK_TYPE):
    if args.lora_load_path:
        config = PeftConfig.from_pretrained(args.lora_load_path)
        peft_model = PeftModel.from_pretrained(model, args.lora_load_path, is_trainable=True, config=config)
    else:
        target_modules = [name for name in args.target_modules.split(',') if name]
        modules_to_save = [name for name in args.modules_to_save.split(',') if name] if args.modules_to_save else None
        config = LoraConfig(
            r=args.lora_rank,
            target_modules=target_modules,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            fan_in_fan_out=args.lora_fan_in_fan_out,
            bias=args.lora_bias,
            modules_to_save=modules_to_save,
            task_type=task_type,
        )
        peft_model = get_peft_model(model, config)

        base_model_name = Path(args.llm_model_name_or_path).name
        peft_model_name = f'{base_model_name}_tm_{"_".join(target_modules)}_rank_{args.lora_rank}_alpha_{args.lora_alpha}_dropout_{args.lora_dropout}_fifo_{args.lora_fan_in_fan_out}_bias_{args.lora_bias}'
        peft_model.save_pretrained(peft_model_name)
        _logger.info(f'Create lora model and save under {peft_model_name}')
    return peft_model
