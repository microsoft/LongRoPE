# python3 main.py \
#     --model hf-causal-experimental \
#     --model_args pretrained=yahma/llama-7b-hf \
#     --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq \
#     --device cuda:0 \
#     --batch_size 1

#!/bin/bash
export PYTHONPATH='.'

base_model=$1 # e.g., decapoda-research/llama-7b-hf
pretrained_path=$2 

python main.py \
    --model lora-pruner \
    --model_args pretrained=$base_model,peft=$pretrained_path \
    --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq,storycloze_2018 \
    --device cuda:0 \
    --output_path results/${tune_id}_$epoch.json \
    --no_cache
# nohup bash eval_harness.sh > eval_log.txt 2>&1 &