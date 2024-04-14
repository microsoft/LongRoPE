# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

# flake8: noqa
import src.criterion_helper
import src.data_helper
import src.model_helper
import src.hf_finetune
from fairseq_cli.train import cli_main

if __name__ == "__main__":
    cli_main()
