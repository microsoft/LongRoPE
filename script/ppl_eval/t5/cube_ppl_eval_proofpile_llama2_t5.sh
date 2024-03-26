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
setting["longrope_128k_mistral"]="--model ${path_team}/ft_out_model/cube-16k-mistral-128k/ck-400 --method longrope --finetuned --factor 32.0 --original_max_position_embeddings 4096 --sliding_window_attention 131072"

# MIS_BOOKS3_256K="--tokenized ${path_dir}/books3-test-sampled-1024k-tokenized-mistral --dataset_min_tokens 2097152 --samples 20 --sliding_window 262144"
# MIS_BOOKS3_2048K="--tokenized ${path_dir}/books3-test-sampled-1024k-tokenized-mistral --dataset_min_tokens 2097152 --samples 20 --sliding_window 1048576"
LA_BOOKS3_256K="--tokenized ${path_dir}/books3-test-sampled-1024k-tokenized --dataset_min_tokens 2097152 --samples 20 --sliding_window 262144"
LA_BOOKS3_2048K="--tokenized ${path_dir}/books3-test-sampled-1024k-tokenized --dataset_min_tokens 2097152 --samples 20 --sliding_window 1048576"

cache_dir="../cache_dir"
output_dir=./evaluation/result

# save_memory="\
# --aggressive_mem_causal_lm \
# --aggressive_mem_decoder \
# --aggressive_mem_attn"
save_memory="" # check

config_list=("longrope_128k")
max_tokens_list=(8192 16384 32768)

gpu_num=1
for config in "${config_list[@]}"; do
    for max_tokens in "${max_tokens_list[@]}"; do
        echo "####### $config, max-tokens=$max_tokens #############"
        CUDA_VISIBLE_DEVICES=0,1,2,3 /mnt/yiran/miniconda3/envs/cube4infer/bin/torchrun \
            --nproc_per_node=$gpu_num \
            --master_port 29520 \
            evaluation/perplexity.py \
            ${PROOFPILE_128k}\
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
    done
done

gpu_num=4
for config in "${config_list[@]}"; do
    for max_tokens in "${max_tokens_list[@]}"; do
        echo "####### $config, max-tokens=$max_tokens #############"
        CUDA_VISIBLE_DEVICES=0,1,2,3 /mnt/yiran/miniconda3/envs/cube4infer/bin/torchrun \
            --nproc_per_node=$gpu_num \
            --master_port 29520 \
            evaluation/perplexity.py \
            ${PROOFPILE_128k}\
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
            --tp_size $gpu_num
    done
done



# max_tokens_list=(262144)
# for config in "${config_list[@]}"; do
#     for max_tokens in "${max_tokens_list[@]}"; do
#         echo "####### $config, max-tokens=$max_tokens #############"
#         python evaluation/perplexity.py \
#             ${PROOFPILE_256k}\
#             ${setting[$config]} \
#             --max_tokens $max_tokens \
#             --min_tokens $max_tokens \
#             --tokens_step 2048 \
#             --output_file "${output_dir}/t5_proofpile_${config}_${max_tokens}.csv" \
#             --original_max_position_embeddings 4096 \
#             --flash_attn \
#             ${save_memory} \
#             --cache_dir $cache_dir
#     done
# done
