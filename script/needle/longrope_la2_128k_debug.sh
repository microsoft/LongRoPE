#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

source ./path_teamdrive.sh
path_dir=$path_team

# model="${path_team}/Llama-2-7b-hf/"

model_128="${path_dir}/ft_out_model/cube-128k-dim-piece-mono-500-#m0/ck-400/"
model_256="${path_dir}/ft_out_model/cube_256k_from_128k/ck-600/"

declare -A setting

# longrope 128k
setting["longrope_128k"]="-m ${model_128} --method longrope --finetuned --factor 32.0"

# longrope 256k
setting["longrope_256k"]="-m ${model_256} --method longrope --finetuned --factor 64.0"

setting["base"]="--model ${path_team}/Llama-2-7b-hf/"
base=${path_team}/Llama-2-7b-hf/

name=longrope_128k

# mkdir -p evaluation/needle/logs evaluation/needle/img evaluation/needle/result
rm ./evaluation/needle/result/longrope_128k_load/*

(
python -u evaluation/needle/needle_in_haystack.py \
    --s_len 0 --e_len 128000 \
    --context_lengths_min 1024 \
    --context_lengths_max 128000 \
    --context_lengths_num_intervals 5 \
    --document_depth_percent_intervals 1 \
    --model_provider LLaMA \
    --model_path ${model_128} \
    --result_path ./evaluation/needle/result/${name}_debug/ \
    ${setting[${name}]} \
    --flash_attn \
    --max_tokens 4000 \

) 2>&1  | tee evaluation/needle/logs/eval_${name}_debug.log

# python evaluation/needle/visualize.py 
