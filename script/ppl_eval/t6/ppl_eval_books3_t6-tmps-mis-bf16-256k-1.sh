#!/bin/bash
export CUDA_VISIBLE_DEVICES=4

# path_dir=/your/path/to/store/model/or/dataset
# your path to model

source ./path_teamdrive.sh
path_dir=$path_team

# model setting
declare -A setting

# longrope 128k
setting["longrope_128k"]="--model ${path_dir}/ft_out_model/cube-128k-dim-piece-mono-500-#m0/ck-400 --method longrope --finetuned --factor 32.0"

# longrope 256k
setting["longrope_256k"]="--model ${path_dir}/ft_out_model/cube_256k_from_128k/ck-600 --method longrope  --finetuned --factor 64.0"

setting["longrope_mistral_128k_bf16"]="--model ${path_team}/ft_out_model/cube-mis-128k-bf16/ck-1_1000 --method longrope  --finetuned --factor 32 --sliding_window_attention 131072 --original_max_position_embeddings 4096"

setting["longrope_mistral_256k_bf16_from_step500"]="--model ${path_team}/ft_out_model/cube-mis-256k-bf16-step-500/ck-1_500 --method longrope --finetuned --factor 64 --sliding_window_attention 262144 --original_max_position_embeddings 4096"


# dataset setting
BOOKS3="--tokenized ${path_team}/books3-test-sampled-1024k-tokenized-mistral --dataset_min_tokens 2097152 --samples 20 --sliding_window 262144"

cache_dir="../cache_dir"
output_dir=./script/ppl_eval/t6

save_memory=""
# save_memory="" # check

# config_list=("longlora" "codellama" "yarn_64k" "yarn_128k" "longrope_128k" "longrope_256k")
config_list=("longrope_mistral_256k_bf16_from_step500" ) # check

echo "dataset BOOKS3 20sample"
# max_tokens_list=(8192 32768 65536 98304 131072 262144 524288 1048576)
# max_tokens_list=(8192 32768 65536 98304 131072 262144) # check
max_tokens_list=(8192) # check
tmps_list=(1.07)

for tmps in "${tmps_list[@]}"; do
    max_tokens=8192
    for config in "${config_list[@]}"; do
        echo "####### tmps $tmps, $config, max-tokens=$max_tokens #############"
        python evaluation/perplexity.py \
            ${BOOKS3}\
            ${setting[$config]} \
            --max_tokens $max_tokens \
            --min_tokens $max_tokens \
            --tokens_step 2048 \
            --output_file "${output_dir}/t6_books3_${config}_${max_tokens}_${tmps}.csv" \
            --flash_attn \
            ${save_memory} \
            --cache_dir $cache_dir \
            --tmps $tmps
    done
done
