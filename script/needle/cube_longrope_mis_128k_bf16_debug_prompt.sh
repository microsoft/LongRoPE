#!/bin/bash
source ./path_teamdrive.sh
path_dir=$path_team

# model="${path_team}/Llama-2-7b-hf/"


# mistral_128k="${path_team}/ft_out_model/cube-16k-mistral-128k/ck-400"
mistral_128k="${path_team}/ft_out_model/cube-mis-128k-bf16/ck-1_1000"
# mistral_256k="${path_team}/ft_out_model/cube-16k-mistral-256k/ck-400"

declare -A setting

# longrope 128k
setting["longrope_128k"]="-m ${mistral_128k} --method longrope --finetuned --factor 32.0 --sliding_window_attention 131072"

# longrope 256k
# setting["longrope_256k"]="-m ${model_256} --method longrope --finetuned --longrope_para /mnt/yiran/2048k-mistral-256k/s-PI/evolution/test/result_alpha/mistral_262144_dim_mono_ppl9.csv --factor 64.0 --sliding_window_attention 262144"

# 检查是否传入了参数  
if [ -z "$1" ]; then  
    echo "Error: No prompt_name provided."  
    echo "Usage: $0 <prompt_name>"  
    exit 1  
fi  

prompt_name=$1
echo "prompt_name: $prompt_name"
# ANTHROPIC_TEMPLATE_REV1


# mkdir -p evaluation/needle/logs evaluation/needle/img evaluation/needle/result

name=longrope_mis_128k_bf16_debug_$prompt_name
rm -rf ./evaluation/needle/result/$name

gpu_num=4
(
CUDA_VISIBLE_DEVICES=4,5,6,7 /mnt/yiran/miniconda3/envs/cube4infer/bin/torchrun \
    --nproc_per_node=$gpu_num \
    --master_port 29510 \
    evaluation/needle/needle_in_haystack.py \
    --s_len 0 --e_len 128000 \
    --context_lengths_min 1024 \
    --context_lengths_max 128000 \
    --context_lengths_num_intervals 10 \
    --document_depth_percent_intervals 5 \
    --model_provider Mistral \
    --model_path ${mistral_128k} \
    --result_path ./evaluation/needle/result/$name/ \
    ${setting["longrope_128k"]} \
    --flash_attn \
    --max_tokens 4000 \
    --prompt_template $prompt_name \
    --use_cube \
    --rope_method s_pi \
    --rope_tmps su \
    --use_cache \
    --tp_size $gpu_num \
    
) 2>&1  | tee evaluation/needle/logs/eval_$name.log

# python evaluation/needle/visualize.py 
