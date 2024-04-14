from os import PathLike
from pathlib import Path
from tqdm import tqdm
from typing import Callable

import torch
import numpy as np

from datasets import load_from_disk
from transformers import DataCollatorForLanguageModeling
from fairseq.data import FairseqDataset

from .utils import IGNORE_IDX, get_tokenizer


class HFDiskDataset(FairseqDataset):
    """
    Load a dataset by ```datasets.load_from_disk```.
    You could preprocess your raw data and save the processed dataset by ```datasets.Dataset.save_to_disk```.
    Each sample of the loaded dataset should have a key `input_ids`.

    Args:
        dataset_path (str | PathLike):
            Path (e.g. `"dataset/train"`) or remote URI (e.g. `"s3://my-bucket/dataset/train"`) of
            the [`Dataset`] or [`DatasetDict`] directory where the dataset will be loaded from.
            Loading the dataset by ```datasets.load_from_disk(data_path)```,
            please make sure your dataset can be load by this way.
        data_collator (Callable | None, defaults to `None`)
            A callable function used to collate a group of samples to a mini batch.
            The mini batch should be a dict with format {'input_ids': ..., 'labels': ...}.

            We assume the data_collator returned value has the same semantics with transformers.DataCollatorForLanguageModeling(..., mlm=False).
            That is, the `labels` should have same value with `input_idx` except the `IGNORE_IDX`.
        tokenizer_name_or_path (str | None, defaults to `None`):
            A huggingface hub id or a local path, for example, `"meta-llama/Llama-2-7b-hf"` or `"/home/USER_NAME/llama2-7b"`
            This tokenizer is used to initialize the default data collator:
                transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
        pad_to_fixed_length (bool, defaults to `True`):
            If pad each sample to fixed length.
        cache_sizes (bool, defaults to `True`):
            Fairseq dataset need to know each sample length.
            If set to True, will save the sizes information of the dataset to `dataset_path/fsds_sizes.np`
    """
    def __init__(self, dataset_path: str | PathLike,
                 data_collator: Callable | None = None,
                 tokenizer_name_or_path: str | None = None,
                 model_max_length: int | None = None,
                 pad_to_fixed_length: bool = True,
                 cache_sizes: bool = True,
                 shuffle: bool = False) -> None:
        self.dataset = load_from_disk(dataset_path)
        sizes_path = Path(dataset_path) / 'fsds_sizes.np'
        if sizes_path.exists():
            self.sizes = torch.load(sizes_path)
        else:
            sizes = []
            for data in tqdm(self.dataset):
                sizes.append(len(data['input_ids']))
            self.sizes = np.array(sizes)
            if cache_sizes:
                torch.save(self.sizes, sizes_path)

        self.tokenizer = get_tokenizer(tokenizer_name_or_path, model_max_length)
        self.pad_to_fixed_length = pad_to_fixed_length
        self.shuffle = shuffle

        if data_collator:
            self.data_collator = data_collator
        else:
            self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        

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

        src_tokens = mini_batch.pop('input_ids')
        if self.pad_to_fixed_length:
            pad_len = self.tokenizer.model_max_length - src_tokens.shape[-1]
            assert pad_len >= 0
            _mini_batch['src_tokens'] = torch.nn.functional.pad(src_tokens, (0, pad_len), 'constant', self.tokenizer.pad_token_id).contiguous()
        else:
            _mini_batch['src_tokens'] = src_tokens

        shift_labels = mini_batch['labels'][..., 1:]
        if self.pad_to_fixed_length:
            pad_len = self.tokenizer.model_max_length - shift_labels.shape[-1]
            assert pad_len >= 0
            _mini_batch['labels'] = torch.nn.functional.pad(shift_labels, (0, pad_len), 'constant', IGNORE_IDX).contiguous()
        else:
            _mini_batch['labels'] = torch.nn.functional.pad(shift_labels, (0, 1), 'constant', IGNORE_IDX).contiguous()

        return {
            "id": torch.LongTensor([s['id'] for s in samples]),
            "nsentences": len(samples),
            "ntokens": sum(self.sizes[s['id']] for s in samples),
            "net_input": _mini_batch,
            "target": _mini_batch.pop('labels'),
        }
