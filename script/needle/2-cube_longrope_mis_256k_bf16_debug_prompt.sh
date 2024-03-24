#!/bin/bash
source ./path_teamdrive.sh
path_dir=$path_team

export TOKENIZERS_PARALLELISM=false

# 检查是否传入了足够的参数  
if [ $# -lt 2 ]; then  
    echo "Error: Insufficient arguments provided."  
    echo "Usage: $0 <ck_step> <prompt_name>"  
    exit 1  
fi  

ck_step=$1
prompt_name=$2
echo "ck_step: $ck_step"
echo "prompt_name: $prompt_name"
# ANTHROPIC_TEMPLATE_REV1


# mistral_128k="${path_team}/ft_out_model/cube-mis-128k-bf16/ck-${ck_step}"
mistral_256k="${path_team}/ft_out_model/cube-mis-256k-bf16/ck-${ck_step}"
echo "cube-mis-256k-bf16 from 1000"

declare -A setting

# longrope 128k
# setting["longrope_128k"]="-m ${mistral_128k} --method longrope --finetuned --factor 32.0 --sliding_window_attention 131072"
setting["longrope_256k"]="-m ${mistral_256k} --method longrope --finetuned --factor 64.0 --sliding_window_attention 262144"

# mkdir -p evaluation/needle/logs evaluation/needle/img evaluation/needle/result
# # clean pt
pt_list="fullmodel.pt.* gencode* cube_graph.pb dist_param_map.pt"

name="2-cube_longrope_mis_256k_bf16_from_1000_ck_${ck_step}_debug_${prompt_name}_needle_new"
rm -rf ./evaluation/needle/result/$name

echo "cube trace ..."
gpu_num=1

rm $pt_list
CUDA_VISIBLE_DEVICES=0 /home/aisilicon/miniconda3/envs/cube4infer/bin/torchrun \
    --nproc_per_node=$gpu_num \
    --master_port 29510 \
    evaluation/needle/needle_in_haystack.py \
    --s_len 0 --e_len 128000 \
    --context_lengths_min 1024 \
    --context_lengths_max 128000 \
    --context_lengths_num_intervals 20 \
    --document_depth_percent_intervals 5 \
    --model_provider Mistral \
    --model_path ${mistral_256k} \
    --result_path ./evaluation/needle/result/$name/ \
    ${setting["longrope_256k"]} \
    --flash_attn \
    --max_tokens 4000 \
    --prompt_template $prompt_name \
    --needle_type "new" \
    --use_cube \
    --rope_method s_pi \
    --rope_tmps su \
    --use_cache \
    --tp_size $gpu_num \
    --cube_trace

echo "cube run ..."
gpu_num=8
(
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 /home/aisilicon/miniconda3/envs/cube4infer/bin/torchrun \
    --nproc_per_node=$gpu_num \
    --master_port 29510 \
    evaluation/needle/needle_in_haystack.py \
    --s_len 0 --e_len 128000 \
    --context_lengths_min 1024 \
    --context_lengths_max 128000 \
    --context_lengths_num_intervals 10 \
    --document_depth_percent_intervals 5 \
    --model_provider Mistral \
    --model_path ${mistral_256k} \
    --result_path ./evaluation/needle/result/$name/ \
    ${setting["longrope_256k"]} \
    --flash_attn \
    --max_tokens 4000 \
    --prompt_template $prompt_name \
    --needle_type "new" \
    --use_cube \
    --rope_method s_pi \
    --rope_tmps su \
    --use_cache \
    --tp_size $gpu_num \
    
) 2>&1  | tee evaluation/needle/logs/eval_$name.log

# python evaluation/needle/visualize.py 
