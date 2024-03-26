#!/bin/bash
source ./path_teamdrive.sh
path_dir=$path_team

export TOKENIZERS_PARALLELISM=false

# 检查是否传入了足够的参数  
# if [ $# -lt 2 ]; then  
#     echo "Error: Insufficient arguments provided."  
#     echo "Usage: $0 <ck_step> <prompt_name>"  
#     exit 1  
# fi  

ck_step=1_500
prompt_name=ANTHROPIC_TEMPLATE_ORIGINAL
echo "ck_step: $ck_step"
echo "prompt_name: $prompt_name"
# ANTHROPIC_TEMPLATE_REV1


# mistral_128k="${path_team}/ft_out_model/cube-mis-128k-bf16/ck-${ck_step}"
mistral_256k="${path_team}/ft_out_model/cube-mis-256k-bf16-step-500/ck-${ck_step}"
# mistral_256k="${path_team}/ft_out_model/longrope-256k-sft-mis/mis-256k-longalphaca-12k/ck-${ck_step}"

echo "cube-mis-256k-bf16-step-500 | needle origin"

declare -A setting

# longrope 128k
# setting["longrope_128k"]="-m ${mistral_128k} --method longrope --finetuned --factor 32.0 --sliding_window_attention 131072"
setting["longrope_256k"]="-m ${mistral_256k} --method longrope --finetuned --factor 64.0"


# 1k, 2k, 4k, 8k, 16k, 32k, 64k, 128k, 200k, 400k, 500k, 800k, 1000k, 1500k, 1800k,  2m
2m_list="1000,2000,4000,8000,16000,64000,128000,200000,400000,500000,800000,10000000,1500000,1800000,2000000"

# mkdir -p evaluation/needle/logs evaluation/needle/img evaluation/needle/result
# # clean pt
pt_list="fullmodel.pt.* gencode* cube_graph* dist_param_map.pt"
python_path=$(which python)
torch_path=$(dirname $python_path)


name="33-2m-cube_longrope_mis_256k_sft_ck_${ck_step}_debug_${prompt_name}_needle_origin"
rm -rf ./evaluation/needle/result/$name

echo "cube trace ..."
gpu_num=1

rm $pt_list
CUDA_VISIBLE_DEVICES=0 $torch_path/torchrun \
    --nproc_per_node=$gpu_num \
    --master_port 29510 \
    evaluation/needle/needle_in_haystack.py \
    --s_len 0 --e_len 256000 \
    --context_lengths_min 1024 \
    --context_lengths_max 256000 \
    --context_lengths_num_intervals 20 \
    --document_depth_percent_intervals 5 \
    --model_provider Mistral \
    --model_path ${mistral_256k} \
    --result_path ./evaluation/needle/result/$name/ \
    ${setting["longrope_256k"]} \
    --flash_attn \
    --max_tokens 4000 \
    --prompt_template $prompt_name \
    --needle_type "origin" \
    --use_cube \
    --rope_method s_pi \
    --rope_tmps su \
    --use_cache \
    --tp_size $gpu_num \
    --cube_trace


echo "cube run ..."
gpu_num=8
(
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $torch_path/torchrun \
    --nproc_per_node=$gpu_num \
    --master_port 29510 \
    evaluation/needle/needle_in_haystack.py \
    --s_len 0 --e_len 2000000 \
    --context_lengths_min 1000 \
    --context_lengths_max 2000000 \
    --seq_series $2m_list \
    --context_lengths_num_intervals 10 \
    --document_depth_percent_intervals 5 \
    --model_provider Mistral \
    --model_path ${mistral_256k} \
    --result_path ./evaluation/needle/result/$name/ \
    ${setting["longrope_256k"]} \
    --flash_attn \
    --max_tokens 4000 \
    --prompt_template $prompt_name \
    --needle_type "origin" \
    --use_cube \
    --rope_method s_pi \
    --rope_tmps su \
    --use_cache \
    --tp_size $gpu_num \
    
) 2>&1  | tee evaluation/needle/logs/eval_$name.log

# python evaluation/needle/visualize.py 

python evaluation/needle/visualize.py --name 33-2m-bf16-256k --path evaluation/needle/result/$name/ck-$ck_step/