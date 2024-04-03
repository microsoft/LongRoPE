import argparse
import copy
from functools import partial
from itertools import chain

from pathlib import Path
from tqdm import tqdm
import torch
import numpy as np

from datasets import load_dataset, load_from_disk
from transformers import DataCollatorForLanguageModeling
from fairseq.data import FairseqDataset

from build_dict import get_tokenizer

IGNORE_IDX = -100


def hf_tokenize_map_fn(dataset, tokenizer, split, feature, seq_len, num_proc=16):
    # split dataset
    dataset = dataset[split] 
    # redpajama only have on split 'train'
    dataset = dataset.remove_columns([x for x in dataset.column_names if x not in [feature]])

    # Add bos eos
    tokenized_dataset = dataset.map(
        lambda example: tokenizer(
            [t + tokenizer.eos_token for t in example[feature]]),
        batched=True,
        num_proc=num_proc,
        remove_columns=[feature],
    )

    # seq_len 128k
    block_size = seq_len 

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {
            k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + block_size]
                for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        return result

    train_dataset = tokenized_dataset.map(
        group_texts, batched=True, num_proc=num_proc,
    )

    return train_dataset

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
        
        if if_tokenize:
            dataset = load_dataset(path=data_name_or_path, cache_dir=cache_dir)
            
            # Redpajama
            # self.dataset = hf_tokenize_map_fn(dataset=dataset, tokenizer=tokenizer, split=data_split, feature=data_feature, seq_len=max_seq_len, num_proc=data_num_proc)
            
            # Long Red
            train_dataset = load_dataset(path=data_name_or_path, cache_dir=cache_dir)
            print(train_dataset)
            print(len(train_dataset['train'][0]['input_ids']))
            from datasets import DatasetDict
            if isinstance(train_dataset, DatasetDict):
                train_dataset = train_dataset["train"]
            if "input_ids" not in train_dataset.column_names:
                raise RuntimeError("Dataset must include an `input_ids` feature")
            if "labels" not in train_dataset.column_names:
                def add_labels(sample):
                    sample["labels"] = copy.deepcopy(sample["input_ids"])
                    return sample
                train_dataset = train_dataset.map(
                    add_labels, desc="Adding labels", num_proc=args.num_proc)
            if "attention_mask" not in train_dataset.column_names:
                def add_attention_mask(sample):
                    sample["attention_mask"] = torch.ones(
                        len(sample["input_ids"]), dtype=torch.int8)
                    return sample
                train_dataset = train_dataset.map(
                    add_attention_mask, desc="Adding attention mask", num_proc=args.num_proc)
            
            self.dataset = train_dataset
            
            self.dataset.save_to_disk(mapped_save_path)
            
            sizes = []
            for data in tqdm(self.dataset):
                sizes.append(len(data['input_ids']))
            self.sizes = np.array(sizes)
            torch.save(self.sizes, Path(mapped_save_path) / 'sizes.np')
        else:
            self.dataset = load_from_disk(mapped_save_path)
            self.sizes = torch.load(Path(mapped_save_path) / 'sizes.np')
        
        self.shuffle = shuffle
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

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
