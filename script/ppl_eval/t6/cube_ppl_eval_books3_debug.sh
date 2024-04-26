#!/bin/bash
# export HF_DATASETS_CACHE="/path/to/store/model"
export HF_DATASETS_CACHE="../cache"

# path_dir=/your/path/to/store/model/or/dataset
# your path to model

source ./path_teamdrive.sh
path_dir=$path_team


# model setting
declare -A setting


# longrope 128k
setting["longrope_128k"]="--model ${path_dir}/ft_out_model/cube-128k-dim-piece-mono-500-#m0/ck-400/ --method longrope --finetuned --factor 32.0 --original_max_position_embeddings 4096"

# longrope Mistral 128k
setting["longrope_128k_mistral"]="--model ${path_dir}/ft_out_model/cube-16k-mistral-128k/ck-400 --method longrope --finetuned --factor 32.0 --original_max_position_embeddings 4096 --sliding_window_attention 131072"


setting["longrope_mistral_128k_bf16"]="--model ${path_team}/ft_out_model/cube-mis-128k-bf16/ck-1_1000 --method longrope  --finetuned --factor 32.0 --sliding_window_attention 131072"

setting["longrope_mistral_256k_bf16_from_step500"]="--model ${path_team}/ft_out_model/cube-mis-256k-bf16-step-500/ck-1_500 --method longrope  --finetuned --factor 64.0 --sliding_window_attention 262144"

BOOKS3_256K="--tokenized ${path_dir}/books3-test-sampled-1024k-tokenized-mistral --dataset_min_tokens 2097152 --samples 20 --sliding_window 262144"

BOOKS3_2048K="--tokenized ${path_dir}/books3-test-sampled-1024k-tokenized-mistral --dataset_min_tokens 2097152 --samples 20 --sliding_window 1048576"

cache_dir="../cache_dir"
output_dir="./script/ppl_eval/t6/result"

save_memory="" # check

config_list=("longrope_128k" "longrope_128k_mistral")
config_list=("longrope_128k_mistral")

# max_tokens_list=(4096 8192 32768 65536 98304 131072)

max_tokens_list=(8192 16384 32768 1048576)
for config in "${config_list[@]}"; do

    gpu_num=1
    max_tokens=8192
    echo "config: $config trace ... ... "
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 /mnt/yiran/miniconda3/envs/cube4infer/bin/torchrun \
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
        --use_cache \
        --tp_size $gpu_num \
        --cube_trace
    
    echo "config: $config run ... ... "
    gpu_num=8
    for max_tokens in "${max_tokens_list[@]}"; do
        echo "####### $config, max-tokens=$max_tokens #############"
        if (( max_tokens <= 262144 )); then  
            dataset=${BOOKS3_256K}  
        else  
            dataset=${BOOKS3_2048K}  
        fi
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 /mnt/yiran/miniconda3/envs/cube4infer/bin/torchrun \
            --nproc_per_node=$gpu_num \
            --master_port 29520 \
            evaluation/perplexity.py \
            $dataset \
            ${setting[$config]} \
            --max_tokens $max_tokens \
            --min_tokens $max_tokens \
            --tokens_step 2048 \
            --output_file "${output_dir}/t6_books_mis_${config}_${max_tokens}.csv" \
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
