#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
# export HF_DATASETS_CACHE="/path/to/store/model"
export HF_DATASETS_CACHE="../cache"
# path_dir=/your/path/to/store/model/or/dataset
# your path to model

source ./path_teamdrive.sh
path_dir=$path_team


# model setting
declare -A setting

# base Mistral-7b
setting["base_mistral"]="--model ${path_dir}/Mistral-7B-v0.1/ --method pi --factor 1.0 --original_max_position_embeddings 4096"

# special: yarn Mistral -> origin postion 8k
# yarn Mistral 64k
setting["yarn_64k_mistral"]="--model ${path_dir}/Yarn-Mistral-7b-64k --method yarn --finetuned --factor 16.0 --original_max_position_embeddings 8192"

# yarn Mistral 128k
setting["yarn_128k_mistral"]="--model ${path_dir}/Yarn-Mistral-7b-128k --method yarn --finetuned --factor 16.0 --original_max_position_embeddings 8192"

# longrope Mistral 128k
setting["longrope_128k_mistral"]="--model ${path_dir}/ft_out_model/cube-16k-mistral-128k/ck-400 --method longrope --finetuned --factor 32.0 --original_max_position_embeddings 4096"

# longrope Mistral 256k
setting["longrope_256k_mistral"]="--model ${path_dir}/ft_out_model/cube-16k-mistral-256k/ck-400 --method longrope  --finetuned --factor 64.0 --original_max_position_embeddings 4096"



# dataset setting
BOOKS3="--tokenized ${path_dir}/books3-test-sampled-1024k-tokenized-mistral --dataset_min_tokens 2097152 --samples 20 --sliding_window 262144"

cache_dir="../cache_dir"
output_dir=./evaluation/result

save_memory="\
--aggressive_mem_causal_lm"
# save_memory="" # check

config_list=("base_mistral" "yarn_64k_mistral" "yarn_128k_mistral" "longrope_128k_mistral" "longrope_256k_mistral")
config_list=("yarn_64k_mistral" "yarn_128k_mistral" "longrope_128k_mistral" "longrope_256k_mistral") # check

echo "dataset BOOKS3 20sample"
max_tokens_list=(8192 32768 65536 98304 131072 262144 524288 1048576)
# max_tokens_list=(4096 8192 32768 65536 98304 131072)
max_tokens_list=(8192 32768 65536 98304 131072 262144)

for max_tokens in "${max_tokens_list[@]}"; do
    for config in "${config_list[@]}"; do
        echo "####### $config, max-tokens=$max_tokens #############"
        python evaluation/perplexity.py \
            ${BOOKS3}\
            ${setting[$config]} \
            --max_tokens $max_tokens \
            --min_tokens $max_tokens \
            --tokens_step 2048 \
            --output_file "${output_dir}/t6_books3_${config}_${max_tokens}.csv" \
            --sliding_window_attention $max_tokens \
            --flash_attn \
            ${save_memory} \
            --cache_dir $cache_dir
    done
done
