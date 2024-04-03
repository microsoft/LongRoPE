# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass, field
from typing import Optional

from fairseq.dataclass import FairseqDataclass
from fairseq.tasks import LegacyFairseqTask, register_task
import torch

from build_dict import get_dict
from hf_dataset import WrappedHFDataset
# from hf_dataset_sft import WrappedHFDataset

logger = logging.getLogger(__name__)


@dataclass
class HFFinetuneConfig(FairseqDataclass):
    """
    This config is similar to LanguageModelingConfig, additional key is bos_word, pad_word, eos_word, unk_word
    """
    tokens_per_sample: int = field(
        default=1024,
        metadata={"help": "max number of tokens per sample for LM dataset"},
    )
    pad_to_fixed_length: Optional[bool] = field(
        default=False, metadata={"help": "pad to fixed length"},
    )
    llm_model_name_or_path: str = field(default='')
    data_name_or_path: str = field(
        default="",
        metadata={"help": "The data path."},
    )
    cache_dir: str = field(
        default="",
        metadata={"help": "The cache path."},
    )
    mapped_save_path: str = field(
        default="",
    )


@register_task("hf_finetune", dataclass=HFFinetuneConfig)
class HFFinetuneTask(LegacyFairseqTask):
    def __init__(self, args, dictionary, output_dictionary=None):
        super().__init__(args)
        self.dictionary = dictionary
        self.output_dictionary = output_dictionary or dictionary

    @classmethod
    def setup_dictionary(cls, args, **kwargs):
        dictionary = get_dict(args.llm_model_name_or_path, args.tokens_per_sample)
        output_dictionary = dictionary
        return (dictionary, output_dictionary)

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        dictionary, output_dictionary = cls.setup_dictionary(args, **kwargs)

        return cls(args, dictionary, output_dictionary)

    def build_model(self, args):
        model = super().build_model(args)
        return model

    def load_dataset(
        self, split: str, epoch=1, combine=False, **kwargs
    ) -> WrappedHFDataset:
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, valid1, test)
        """
        if 'train' not in self.datasets:
            self.datasets['train'] = WrappedHFDataset(
                data_name_or_path=self.args.data_name_or_path,
                cache_dir=self.args.cache_dir,
                tokenizer_id=self.args.llm_model_name_or_path,
                max_seq_len=self.args.tokens_per_sample,
                mapped_save_path=self.args.mapped_save_path,
                if_tokenize=False,
                shuffle=True)

        if split in self.datasets:
            pass
        else:
            dataset, _ = torch.utils.data.random_split(self.datasets['train'], [0.01, 0.99])
            self.datasets[split] = dataset

    @property
    def source_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.dictionary

    @property
    def target_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.output_dictionary
