#!/bin/bash
source ./path_teamdrive.sh
path_dir=$path_team

export TOKENIZERS_PARALLELISM=false

# 检查是否传入了足够的参数  
if [ $# -lt  ]; then  
    echo "Error: Insufficient arguments provided."  
    echo "Usage: $0 <model_sft> "  
    exit 1  
fi  

model_sft=$1
# ck_step=$2
prompt_name=ANTHROPIC_TEMPLATE_ORIGINAL
echo "ck_step: $ck_step"
echo "prompt_name: $prompt_name"
# ANTHROPIC_TEMPLATE_REV1

# ck_step
key=$1
echo "key $key"
# 检查$key是否以"700"结尾  
if [[ "$key" == *"1000" ]]; then  
    ck_step="6_1000"  
elif [[ "$key" == *"700" ]]; then  
    ck_step="4_700"  
elif [[ "$key" == *"300" ]]; then  
    ck_step="2_300" 
else  
    echo "e: $ck_step"
    exit 0
fi  

echo $ck_step  


declare -A sft_setting  
  
sft_setting[mis_128k_sft_1000]="${path_team}/sft-mistral/mistral7b_8xH100_1node_sft-mistral-128k-bf16/ck-6_1000/"  
sft_setting[mis_128k_sft_700]="${path_team}/sft-mistral/mistral7b_8xH100_1node_sft-mistral-128k-bf16/ck-4_700/"  
sft_setting[mis_128k_sft_300]="${path_team}/sft-mistral/mistral7b_8xH100_1node_sft-mistral-128k-bf16/ck-2_300/"  
sft_setting[mis_256k_sft_1_1000]="${path_team}/sft-mistral/mistral7b_8xH100_1node_sft-mistral-256k-bf16_1/ck-6_1000/"  
sft_setting[mis_256k_sft_1_700]="${path_team}/sft-mistral/mistral7b_8xH100_1node_sft-mistral-256k-bf16_1/ck-4_700/"  
sft_setting[mis_256k_sft_1_300]="${path_team}/sft-mistral/mistral7b_8xH100_1node_sft-mistral-256k-bf16_1/ck-2_300/"  
sft_setting[mis_256k_sft_2_1000]="${path_team}/sft-mistral/mistral7b_8xH100_1node_sft-mistral-256k-bf16_2/ck-6_1000/"  
sft_setting[mis_256k_sft_2_700]="${path_team}/sft-mistral/mistral7b_8xH100_1node_sft-mistral-256k-bf16_2/ck-4_700/"  
sft_setting[mis_256k_sft_2_300]="${path_team}/sft-mistral/mistral7b_8xH100_1node_sft-mistral-256k-bf16_2/ck-2_300/"  


# mistral_128k="${path_team}/ft_out_model/cube-mis-128k-bf16/ck-${ck_step}"
# mistral_256k="${path_team}/ft_out_model/cube-mis-256k-bf16-step-500/ck-${ck_step}"
# mistral_256k="${path_team}/ft_out_model/cube-mis-256k-bf16/ck-${ck_step}"
# mistral_256k="${path_team}/ft_out_model/longrope-256k-sft-mis/mis-256k-longalphaca-12k/ck-${ck_step}"

echo "cube-$model_sft | needle origin"

declare -A setting


if [[ "$model_sft" == *"128"* ]]; then  
    setting["$model_sft"]="-m ${sft_setting["$model_sft"]} --method longrope --finetuned --factor 32.0 "
else  
    setting["$model_sft"]="-m ${sft_setting["$model_sft"]} --method longrope --finetuned --factor 64.0 "
fi  


# 1000,2000,4000,8000,16000,64000,128000,200000,400000,500000,800000,10000000,1500000,1800000,2000000


# # clean pt
pt_list="fullmodel.pt.* gencode* cube_graph* dist_param_map.pt"
python_path=$(which python)
torch_path=$(dirname $python_path)

nums=22
name="${nums}-cube_longrope_${model_sft}_needle_origin"
rm -rf ./evaluation/needle/result/$name

# echo "{sft_setting["model_sft"]}: ${sft_setting["$model_sft"]}"
# echo "{setting["model_sft"]}: ${setting["$model_sft"]}"
# exit 0


echo "cube trace ..."
gpu_num=1

rm $pt_list
CUDA_VISIBLE_DEVICES=0 ${torch_path}/torchrun \
    --nproc_per_node=$gpu_num \
    --master_port 29510 \
    evaluation/needle/needle_in_haystack.py \
    --s_len 0 --e_len 256000 \
    --context_lengths_min 1024 \
    --context_lengths_max 256000 \
    --context_lengths_num_intervals 20 \
    --document_depth_percent_intervals 5 \
    --model_provider Mistral \
    --model_path ${sft_setting["$model_sft"]} \
    --result_path ./evaluation/needle/result/$name/ \
    ${setting["$model_sft"]} \
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
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ${torch_path}/torchrun \
    --nproc_per_node=$gpu_num \
    --master_port 29510 \
    evaluation/needle/needle_in_haystack.py \
    --s_len 0 --e_len 2000000 \
    --context_lengths_min 1000 \
    --context_lengths_max 2000000 \
    --seq_series "1000,2000,4000,8000,16000,64000,128000,200000,400000,500000,800000" \
    --context_lengths_num_intervals 10 \
    --document_depth_percent_intervals 5 \
    --model_provider Mistral \
    --model_path ${sft_setting["$model_sft"]} \
    --result_path ./evaluation/needle/result/$name/ \
    ${setting["$model_sft"]} \
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

python evaluation/needle/visualize.py --name ${nums}-${model_sft}-origin --path evaluation/needle/result/$name/ck-$ck_step/