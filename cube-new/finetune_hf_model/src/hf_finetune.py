# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass, field

from fairseq.dataclass import FairseqDataclass
from fairseq.tasks import LegacyFairseqTask, register_task
import torch

from .data_helper import HFDiskDataset, InstructionDataset, get_dict, WrappedHFDataset

logger = logging.getLogger(__name__)


from dataclasses import dataclass, field
from typing import Optional

from fairseq.data.indexed_dataset import get_available_dataset_impl
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.tasks import LegacyFairseqTask, register_task
from omegaconf import II


@dataclass
class HFFinetuneConfig(FairseqDataclass):
    """
    This config is similar to LanguageModelingConfig.
    """
    llm_model_name_or_path: str = field(default='', metadata={'help': 'Huggingface model id or path, i.e., meta-llama/Llama-2-7b-hf or /home/USER/llama2'})
    tokens_per_sample: int = field(default=1024, metadata={"help": "max number of tokens per sample for LM dataset"})
    pad_to_fixed_length: bool = field( default=False, metadata={"help": "pad to fixed length"})

    dataset_type: str = field(default="hf_disk", metadata={"help": "The huggingface dataset id or path."})
    # used by all dataset
    dataset_path: str = field(default="", metadata={"help": "The huggingface dataset id or path."})
    # used by instruction
    prompt_type: str = field(default="alpaca", metadata={"help": "The prompt type to use, by default to use the alpaca prompt."})

    cache_dir: str = field(default="", metadata={"help": "The cache path."},)
    mapped_save_path: str = field(default="")

    # TODO common vars below add to parent
    seed: int = II("common.seed")
    batch_size: Optional[int] = II("dataset.batch_size")
    batch_size_valid: Optional[int] = II("dataset.batch_size_valid")
    dataset_impl: Optional[ChoiceEnum(get_available_dataset_impl())] = II(
        "dataset.dataset_impl"
    )
    data_buffer_size: int = II("dataset.data_buffer_size")
    tpu: bool = II("common.tpu")
    use_plasma_view: bool = II("common.use_plasma_view")
    plasma_path: str = II("common.plasma_path")


@register_task("hf_finetune", dataclass=HFFinetuneConfig)
class HFFinetuneTask(LegacyFairseqTask):
    _supported_dataset_type = ['hf_disk', 'instruction', 'longseq', 'test']

    def __init__(self, args, dictionary, output_dictionary=None):
        super().__init__(args)
        self.dictionary = dictionary
        self.output_dictionary = output_dictionary or dictionary

    @classmethod
    def setup_dictionary(cls, args, **kwargs):
        dictionary = get_dict(args.llm_model_name_or_path)
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
        self,
        split: str,
        combine: bool = False,
        task_cfg: FairseqDataclass = None,
        **kwargs
    ):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, valid1, test)
        """
        if 'train' not in self.datasets:
            if self.args.dataset_type not in self._supported_dataset_type:
                raise RuntimeError(f'Task get not supported dataset type {self.args.dataset_type}, the supported type is {self._supported_dataset_type}')
            if self.args.dataset_type == 'hf_disk':
                dataset = HFDiskDataset(dataset_path=self.args.dataset_path,
                                        tokenizer_name_or_path=self.args.llm_model_name_or_path,
                                        model_max_length=self.args.tokens_per_sample,
                                        pad_to_fixed_length=self.args.pad_to_fixed_length,
                                        shuffle=True)
            if self.args.dataset_type == 'instruction':
                dataset = InstructionDataset(instruction_file_path=self.args.dataset_path,
                                             tokenizer_name_or_path=self.args.llm_model_name_or_path,
                                             model_max_length=self.args.tokens_per_sample,
                                             pad_to_fixed_length=self.args.pad_to_fixed_length,
                                             prompt_type=self.args.prompt_type,
                                             shuffle=True)
            if self.args.dataset_type == 'longseq':
                dataset = WrappedHFDataset(data_name_or_path=self.args.dataset_path,
                                           cache_dir=self.args.cache_dir,
                                           tokenizer_id=self.args.llm_model_name_or_path,
                                           max_seq_len=self.args.tokens_per_sample,
                                           mapped_save_path=self.args.mapped_save_path,
                                           if_tokenize=False,
                                           shuffle=True)
            self.datasets['train'] = dataset

        if split in self.datasets:
            pass
        else:
            dataset, _ = torch.utils.data.random_split(self.datasets['train'], [0.01, 0.99])
            # we don't apply validation in finetuning, but fairseq must load validation dataset, so here we split a small dataset as a placeholder.
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
