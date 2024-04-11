#!/bin/bash

# conda create -n longrope_hf_eval python==3.10.13
# conda activate longrope_hf_eval
cd evaluation/lm-evaluation-harness/
pip install -e .
cd ../../
pip install -r requirements.txt
pip install -U datasets