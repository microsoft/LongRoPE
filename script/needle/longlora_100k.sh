#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

source ./path_teamdrive.sh
path_dir=$path_team

HF_HOME=/mnt/yiran/cache/

# model="${path_team}/Llama-2-7b-hf/"
model="Yukang/Llama-2-7b-longlora-100k-ft"
# /mnt/yiran/LongRoPE/evaluation/needle
mkdir -p evaluation/needle/logs evaluation/needle/img evaluation/needle/results

(
python -u evaluation/needle/needle_in_haystack.py \
    --s_len 0 --e_len 31750 \
    --context_lengths_min 1024 \
    --context_lengths_max 31750 \
    --context_lengths_num_intervals 10 \
    --model_provider LLaMA \
    --model_path ${model} \
    --result_path ./evaluation/needle/result/longlora_100k/
    # --use_cache true \

) 2>&1  | tee -a evaluation/needle/logs/eval_longlora_100k-31k.log

# python evaluation/needle/visualize.py 
