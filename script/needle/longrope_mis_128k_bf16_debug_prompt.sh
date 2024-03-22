#!/bin/bash

# 检查是否传入了足够的参数  
if [ $# -lt 3 ]; then  
    echo "Error: Insufficient arguments provided."  
    echo "Usage: $0 <CUDA_VISIBLE_DEVICES> <ck_step> <prompt_name>"  
    exit 1  
fi  

export CUDA_VISIBLE_DEVICES=$1

source ./path_teamdrive.sh
path_dir=$path_team

# model="${path_team}/Llama-2-7b-hf/"

ck_step=$2 
# mistral_128k="${path_team}/ft_out_model/cube-16k-mistral-128k/ck-400"
mistral_128k="${path_team}/ft_out_model/cube-mis-128k-bf16/ck-${ck_step}"
# mistral_256k="${path_team}/ft_out_model/cube-16k-mistral-256k/ck-400"

declare -A setting

# longrope 128k
setting["longrope_128k"]="-m ${mistral_128k} --method longrope --finetuned --factor 32.0 --sliding_window_attention 131072"

# longrope 256k
# setting["longrope_256k"]="-m ${model_256} --method longrope --finetuned --longrope_para /mnt/yiran/2048k-mistral-256k/s-PI/evolution/test/result_alpha/mistral_262144_dim_mono_ppl9.csv --factor 64.0 --sliding_window_attention 262144"


prompt_name=$3
echo "##############################"
echo -e "CUDA_VISIBLE_DEVICES:$1 \nck_step:$2 \nprompt_name: $3\n"
echo "##############################"
# ANTHROPIC_TEMPLATE_REV1


# mkdir -p evaluation/needle/logs evaluation/needle/img evaluation/needle/result

name=longrope_mis_128k_bf16_${ck_step}_debug_${prompt_name}_n
rm -rf ./evaluation/needle/result/$name

(
python -u evaluation/needle/needle_in_haystack.py \
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
    
) 2>&1  | tee evaluation/needle/logs/eval_$name.log

# python evaluation/needle/visualize.py 
