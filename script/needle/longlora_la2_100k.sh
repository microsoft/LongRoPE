#!/bin/bash
export CUDA_VISIBLE_DEVICES=3

source ./path_teamdrive.sh
path_dir=$path_team

model="${path_dir}/Llama-2-7b-longlora-100k-ft/"

# 1k - 128k 40: 127k/40
# 1k - 31750 10: 127k/40

# /mnt/yiran/LongRoPE/evaluation/needle
# mkdir -p evaluation/needle/logs evaluation/needle/img evaluation/needle/results


(
python -u evaluation/needle/needle_in_haystack.py \
    --s_len 0 --e_len 31750 \
    --model_provider LLaMA \
    --result_path ./evaluation/needle/result/longlora-100k/ \
    --model_path ${model} \
    --method pi --factor 8.0 \
) 2>&1  | tee -a evaluation/needle/logs/eval_longlora-100k.log

# python visualize.py 
