#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

source ./path_teamdrive.sh
path_dir=$path_team

model="${path_team}/Llama-2-7b-hf/"



mkdir -p evaluation/needle/logs evaluation/needle/img evaluation/needle/results

(
python -u evaluation/needle/needle_in_haystack.py \
    --s_len 0 --e_len 8192 \
    --context_lengths_min 1024 \
    --context_lengths_max 8192 \
    --context_lengths_num_intervals 10 \
    --model_provider LLaMA \
    --model_path ${model} \
    --use_cache true \

) 2>&1  | tee -a evaluation/needle/logs/eval_llama-2-7b-8k.log

python evaluation/needle/visualize.py 
