#!/bin/bash
# export HF_DATASETS_CACHE="/path/to/store/model"
export HF_DATASETS_CACHE="../cache"

# path_dir=/your/path/to/store/model/or/dataset
# your path to model

source ./path_teamdrive.sh
path_dir=$path_team


# model setting
declare -A setting

# longrope Mistral 128k
# setting["longrope_128k_mistral"]="--model ${path_dir}/ft_out_model/cube-16k-mistral-128k/ck-400 --method longrope --finetuned --factor 32.0 --original_max_position_embeddings 4096 --sliding_window_attention 131072"

# longrope Mistral 256k
# setting["longrope_256k_mistral"]="--model ${path_dir}/ft_out_model/cube-16k-mistral-256k/ck-400 --method longrope  --finetuned --factor 64.0 --original_max_position_embeddings 4096 --sliding_window_attention 262144"

# setting["longrope_mistral_128k_bf16"]="--model ${path_team}/ft_out_model/cube-mis-128k-bf16/ck-1_1000 --method longrope  --finetuned --factor 32.0 --sliding_window_attention 131072"

# setting["longrope_mistral_256k_bf16_from_step500"]="--model ${path_team}/ft_out_model/cube-mis-256k-bf16-step-500/ck-1_500 --method longrope  --finetuned --factor 64.0 --sliding_window_attention 262144"

# setting["longrope_mistral_256k_bf16_from_step1000"]="--model ${path_team}/ft_out_model/cube-mis-256k-bf16/ck-1_500/ --method longrope  --finetuned --factor 64.0 --sliding_window_attention 262144"


# la2 pose 512k
setting["longrope_128k_pose_512k_dy_scale"]="--model ${path_team}/ft_out_model/longrope-la2-128k-pose-512k/cube-la2-128k-pose-512k/ck-1_600/ --method longrope  --finetuned --factor 128.0 "

# setting["longrope_128k_pose_512k_static_scale"]="--model ${path_team}/ft_out_model/longrope-la2-128k-pose-512k/cube-la2-128k-pose-512k/ck-1_600/ --method longrope  --finetuned --factor 128.0 "

# dataset setting
# PROOFPILE_test="--tokenized ${path_team}/proofpile-test-tokenized --dataset_min_tokens 131072 --samples 1 --truncate"

# PROOFPILE_128k="--tokenized ${path_team}/proofpile-test-tokenized --dataset_min_tokens 131072 --samples 10 --truncate"
# PROOFPILE_256k="--tokenized ${path_team}/proofpile-test-tokenized --dataset_min_tokens 262144 --samples 10 --truncate"

BOOKS3_256K="--tokenized ${path_dir}/books3-test-sampled-1024k-tokenized --dataset_min_tokens 2097152 --samples 20 --sliding_window 262144"

BOOKS3_2048K="--tokenized ${path_dir}/books3-test-sampled-1024k-tokenized --dataset_min_tokens 2097152 --samples 20 --sliding_window 1048576"

cache_dir="/mnt/yiran/cache_dir"
output_dir=./script/ppl_eval/t6/result

# # clean pt
pt_list="fullmodel.pt.* gencode* cube_graph* dist_param_map.pt"
# rm $pt_list

# save_memory="\
# --aggressive_mem_causal_lm \
# --aggressive_mem_decoder \
# --aggressive_mem_attn"
save_memory="" # check

# config_list=("base" "together" "longlora" "codellama" "yarn_64k" "yarn_128k" "longrope_128k" "longrope_256k")
# config_list=("base" "together" "longlora" "yarn_64k" "yarn_128k" "longrope_128k" "longrope_256k")

# config_list=("codellama") # check

# config_list=("longrope_128k_mistral" "longrope_mistral_128k_bf16" "longrope_mistral_256k_bf16_from_step500")

config_list=("longrope_128k_pose_512k_dy_scale")
# config_list=()
python_path=$(which python)
torch_path=$(dirname $python_path)
# max_tokens_list=(4096 8192 32768 65536 98304 131072)

max_tokens_list=(8192 131072 262144 524288 1048576)
# max_tokens_list=(1048576)
for config in "${config_list[@]}"; do

    gpu_num=1
    max_tokens=8192
    echo "config: $config trace ... ... "
    rm $pt_list
    CUDA_VISIBLE_DEVICES=0 $torch_path/torchrun \
        --nproc_per_node=$gpu_num \
        --master_port 29520 \
        evaluation/perplexity.py \
        ${BOOKS3_256K}\
        ${setting[$config]} \
        --max_tokens $max_tokens \
        --min_tokens $max_tokens \
        --tokens_step 2048 \
        --output_file "${output_dir}/t5_proofpile_${config}_${max_tokens}.csv" \
        --original_max_position_embeddings 4096 \
        --flash_attn \
        ${save_memory} \
        --cache_dir $cache_dir \
        --use_cube \
        --rope_method s_pi \
        --rope_tmps su \
        --tp_size $gpu_num \
        --cube_trace
    
    echo "config: $config run ... ... "
    gpu_num=8
    for max_tokens in "${max_tokens_list[@]}"; do
        rm -rf /tmp/tmp*
        echo "####### $config, max-tokens=$max_tokens #############"
        if (( max_tokens <= 262144 )); then  
            dataset=${BOOKS3_256K}  
        else  
            dataset=${BOOKS3_2048K}  
        fi
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $torch_path/torchrun \
            --nproc_per_node=$gpu_num \
            --master_port 29520 \
            evaluation/perplexity.py \
            $dataset \
            ${setting[$config]} \
            --max_tokens $max_tokens \
            --min_tokens $max_tokens \
            --tokens_step 2048 \
            --output_file "${output_dir}/7-t6_books_${config}_${max_tokens}.csv" \
            --original_max_position_embeddings 4096 \
            --flash_attn \
            ${save_memory} \
            --cache_dir $cache_dir \
            --use_cube \
            --rope_method s_pi \
            --rope_tmps su \
            --tp_size $gpu_num
    done
done
