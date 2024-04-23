import copy
import json
import logging
from os import PathLike
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Sequence, Callable

import numpy as np
import torch
from transformers import PreTrainedTokenizer

from fairseq.data import FairseqDataset

from .utils import IGNORE_IDX, get_tokenizer

_logger = logging.getLogger(__name__)

# alpaca prompt
ALPACA_PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

# {prompt_type: {"prompt_input": ..., "prompt_no_input": ...}}
PROMPT_MAP = {
    "alpaca": ALPACA_PROMPT_DICT
}


def _tokenize_fn(strings: Sequence[str], tokenizer: PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            # add_special_tokens=False,
            return_tensors="pt",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in tqdm(strings)
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len, label_len in zip(labels, sources_tokenized["input_ids_lens"], examples_tokenized["labels_lens"]):
        label[:source_len] = IGNORE_IDX
        label[label_len:] = IGNORE_IDX
    sizes = examples_tokenized["labels_lens"]
    return input_ids, labels, sizes


class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    def __init__(self, tokenizer: PreTrainedTokenizer, pad_to_fixed_length: bool = True):
        self.tokenizer = tokenizer
        self.pad_to_fixed_length = pad_to_fixed_length

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # NOTE: (nishang) I don't know way, but if the sample is end of eos token, the model will easy to get nan during forward.
        for instance in instances:
            if instance["input_ids"][instance["ntokens"] - 1] == self.tokenizer.eos_token_id:
                instance["input_ids"][instance["ntokens"] - 1] = self.tokenizer.pad_token_id

        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_IDX)

        if self.pad_to_fixed_length:
            assert self.tokenizer.model_max_length >= input_ids.shape[-1]
            input_ids = torch.nn.functional.pad(input_ids, (0, self.tokenizer.model_max_length-input_ids.shape[-1]), value=self.tokenizer.pad_token_id).contiguous()
            assert self.tokenizer.model_max_length >= labels.shape[-1]
            labels = torch.nn.functional.pad(labels, (0, self.tokenizer.model_max_length-labels.shape[-1]), value=IGNORE_IDX).contiguous()
        input_ids = torch.nn.functional.pad(input_ids[..., :-1], (1, 0), "constant", self.tokenizer.eos_token_id).contiguous()

        return {
            "id": torch.LongTensor([s["id"] for s in instances]),
            "nsentences": len(instances),
            "ntokens": sum(s["ntokens"] for s in instances),
            "net_input": {
                "src_tokens": input_ids,
            },
            "target": labels,
        }


class InstructionDataset(FairseqDataset):
    def __init__(self, instruction_file_path: str | PathLike,
                 data_collator: Callable | None = None,
                 tokenizer_name_or_path: str | None = None,
                 model_max_length: int | None = None,
                 pad_to_fixed_length: bool = True,
                 prompt_type: str = "alpaca",
                 shuffle: bool = False) -> None:
        file_path = Path(instruction_file_path)
        if not file_path.is_file():
            raise RuntimeError(f"{instruction_file_path} is not a file path, or it is not exist.")

        tokenizer = get_tokenizer(tokenizer_name_or_path, model_max_length)

        cache_dir = file_path.parent / f"{file_path.stem}_cube_data_cache_seqlen_{model_max_length}"

        if cache_dir.exists():
            self.input_ids = torch.load(cache_dir / 'input_ids.pth')
            self.labels = torch.load(cache_dir / 'labels.pth')
            self.sizes = torch.load(cache_dir / 'sizes.pth')
        else:
            with file_path.open() as f:
                if file_path.suffix == ".json":
                    examples = json.load(f)
                else:
                    examples = [json.loads(line) for line in f.readlines()]

            if prompt_type not in PROMPT_MAP:
                raise RuntimeError(f"Unsupported prompt type: {prompt_type}")
            prompt_input, prompt_no_input = PROMPT_MAP[prompt_type]["prompt_input"], PROMPT_MAP[prompt_type]["prompt_no_input"]

            sources = [
                prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
                for example in examples
            ]
            targets = [f"{example['output']}{tokenizer.eos_token}" for example in examples]

            self.input_ids, self.labels, sizes = preprocess(sources, targets, tokenizer)
            self.sizes = np.array(sizes)

            cache_dir.mkdir(parents=True, exist_ok=True)
            torch.save(self.input_ids, cache_dir / 'input_ids.pth')
            torch.save(self.labels, cache_dir / 'labels.pth')
            torch.save(self.sizes, cache_dir / 'sizes.pth')

        if data_collator:
            self.data_collator = data_collator
        else:
            self.data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, pad_to_fixed_length=pad_to_fixed_length)
        
        self.shuffle = shuffle

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return {
            "id": i,
            "ntokens": self.sizes[i],
            "input_ids": self.input_ids[i],
            "labels": self.labels[i],
        }

    def size(self, index):
        return self.sizes[index]

    def num_tokens(self, index):
        return self.sizes[index]

    def ordered_indices(self):
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]
        order.append(self.sizes)
        return np.lexsort(order)

    def collater(self, samples):
        if len(samples) == 0:
            return {}

        return self.data_collator(samples)
