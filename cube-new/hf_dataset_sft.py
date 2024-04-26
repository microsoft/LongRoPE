import argparse
from functools import partial
from itertools import chain

from pathlib import Path
from tqdm import tqdm
import torch
import numpy as np

from datasets import load_dataset, load_from_disk
import datasets
from transformers import DataCollatorForLanguageModeling
from fairseq.data import FairseqDataset

from build_dict import get_tokenizer

# sft
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import transformers
from torch.utils.data import Dataset
import logging
import io
import os
import copy
import json
import math
from tqdm import tqdm

IGNORE_IDX = -100
IGNORE_INDEX = -100
def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

PROMPT_DICT = {
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
    "prompt_no_input_llama2":(
        "[INST] <<SYS>>\n"
        "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\n"
        "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n"
        "<</SYS>> \n\n {instruction} [/INST]"
    ),
    "prompt_input_llama2": (
        "[INST] <<SYS>>\n"
        "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\n"
        "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n"
        "<</SYS>> \n\n {instruction} \n{input} [/INST]"
    ),
    "prompt_llama2": "[INST]{instruction}[/INST]"
}

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
            
        )
        for text in tqdm(strings, desc="Tokenizing")
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
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = jload(data_path)

        logging.warning("Formatting inputs...")

        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input_llama2"], PROMPT_DICT["prompt_llama2"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]

        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)
        
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])
    
    def save_to_disk(self, mapped_save_path):
        
        data = dict(
            input_ids=self.input_ids,
            labels=self.labels,
            # attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        
        data = datasets.Dataset.from_dict(data).with_format("torch")
        
        print(data)
        print(data[0])
        data.save_to_disk(mapped_save_path)
        

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


class WrappedHFDataset(FairseqDataset):
    def __init__(self, data_name_or_path: str,
                 cache_dir: str,
                 tokenizer_id: str,
                 max_seq_len: int,
                 mapped_save_path: str,
                 if_tokenize: bool = False,
                 shuffle: bool = False,
                 data_split: str = 'train',
                 data_feature: str = 'text',
                 data_num_proc: int = 16,
                 ) -> None:
        tokenizer = get_tokenizer(tokenizer_id, max_seq_len)
        print("tokenizer.pad_token_id", tokenizer.pad_token_id)
        print("tokenizer.model_max_length", tokenizer.model_max_length)
        
        if if_tokenize:
            
            # sft
            print("begin sft data")
            """Make dataset and collator for supervised fine-tuning."""
            train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_name_or_path)
            print(train_dataset[0])
            self.dataset = train_dataset
            
            sizes = []
            for data in tqdm(self.dataset):
                sizes.append(min(len(data['input_ids']), max_seq_len))
            self.sizes = np.array(sizes)
            torch.save(self.sizes, Path(mapped_save_path) / 'sizes.np')
            
            self.dataset.save_to_disk(mapped_save_path)
            # torch.save(self.dataset, mapped_save_path + '/dataset.pt')
            
        else:
            self.dataset = load_from_disk(mapped_save_path)
            self.sizes = torch.load(Path(mapped_save_path) / 'sizes.np')
        
        
        # # padding 
        # self.dataset = load_from_disk(mapped_save_path)
        print(self.dataset[0])
        input_ids, labels = self.dataset['input_ids'], self.dataset['labels']
        print("$before pad")
        print("input_ids[0][:5]+input_ids[0][-5:]", input_ids[0][:5], input_ids[0][-5:])
        print("labels[0][:5]+labels[0][-5:]", labels[0][:5], labels[0][-5:])
        # input_ids[0][1:] = input_ids[0][:-1].clone()
        # input_ids[0][0] = tokenizer.bos_token_id
        
        # print("$after shift")
        # print("input_ids[0][:5]+input_ids[0][-5:]", input_ids[0][:5], input_ids[0][-5:])
        # print("labels[0][:5]+labels[0][-5:]", labels[0][:5], labels[0][-5:])
        # exit(0)
        
        for i in range(self.dataset.num_rows):
            # 如果长度不足16384，则进行填充
            # input_ids[i]
            if len(input_ids[i]) < max_seq_len:
                input_ids[i] = torch.cat([input_ids[i], torch.full((max_seq_len - len(input_ids[i]),), tokenizer.pad_token_id)], dim=0)
                labels[i] = torch.cat([labels[i], torch.full((max_seq_len - len(labels[i]),), IGNORE_INDEX)], dim=0)
            
            # 如果长度超过16384，则进行截断
            elif len(input_ids[i]) > max_seq_len:
                input_ids[i] = input_ids[i][:max_seq_len]
                labels[i] = labels[i][:max_seq_len]
        
        # print("$before shift")
        # print("input_ids[0][:5]+input_ids[0][8245-5:8245+5]", input_ids[0][:5]+input_ids[0][8245-5:8245+5])
        # print("labels[0][:5]+labels[0][8245-5:8245+5]", labels[0][:5]+labels[0][8245-5:8245+5])
        # input_ids[0][1:] = input_ids[0][:-1]
        # input_ids[0][0] = tokenizer.bos_token_id
        
        print("$after pad")
        print("input_ids[0][:5]+input_ids[0][8245-5:8245+5]", input_ids[0][:5], input_ids[0][8245-5:8245+5])
        print("labels[0][:5]+labels[0][8245-5:8245+5]", labels[0][:5], labels[0][8245-5:8245+5])
        # exit(0)
        
        
        self.dataset = datasets.Dataset.from_dict({'input_ids': input_ids, 'labels': labels}).with_format("torch")
        
        # self.dataset['input_ids'].sh
        print("after padding")
        for i in range(0, len(self.dataset), 1000):
            print(i, self.dataset[i]['input_ids'].shape)
        
        # resize data len
        for i in tqdm(range(self.sizes.shape[0])):
            if self.sizes[i] > max_seq_len:
                print(f"data row {i} > {max_seq_len}")
                self.sizes[i] == max_seq_len
            
                
        self.shuffle = shuffle
        # self.data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        # pad in cube
        self.data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        """id is needed in fairseq"""
        data = self.dataset.__getitem__(int(i))
        data['id'] = i
        return data

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
        mini_batch = self.data_collator(samples)
        _mini_batch = {}
        _mini_batch['src_tokens'] = mini_batch.pop('input_ids')
        
        _mini_batch['labels'] = torch.nn.functional.pad(mini_batch['labels'][..., 1:], (0, 1), 'constant', IGNORE_IDX).contiguous()
        # _mini_batch['attention_mask'] = mini_batch['attention_mask']
        # _mini_batch['id'] = mini_batch['id']

        return {
            "id": torch.LongTensor([s['id'] for s in samples]),
            "nsentences": len(samples),
            "ntokens": sum(self.sizes[s['id']] for s in samples),
            "net_input": _mini_batch,
            "target": _mini_batch.pop('labels'),
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name_or_path')
    parser.add_argument('--cache_dir')
    parser.add_argument('--tokenizer_id')
    parser.add_argument('--max_seq_len')
    parser.add_argument('--mapped_save_path')
    parser.add_argument('--if_tokenize')
    parser.add_argument('--data_split')
    parser.add_argument('--data_feature')
    parser.add_argument('--data_num_proc')
    
    args = parser.parse_args()
    print("if_tokenize", args.if_tokenize)
    WrappedHFDataset(data_name_or_path=args.data_name_or_path,
                     cache_dir=args.cache_dir,
                     tokenizer_id=args.tokenizer_id,
                     max_seq_len=int(args.max_seq_len),
                     mapped_save_path=args.mapped_save_path,
                     if_tokenize=args.if_tokenize,
                     shuffle=True,
                    #  data_split=args.data_split if args.data_split!=None else None,
                    #  data_feature=args.data_feature,
                    #  data_num_proc=args.data_num_proc
                     )
