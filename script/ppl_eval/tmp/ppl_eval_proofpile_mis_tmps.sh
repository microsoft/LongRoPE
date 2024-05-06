#!/bin/bash
export CUDA_VISIBLE_DEVICES=5
# export HF_DATASETS_CACHE="/path/to/store/model"
export HF_DATASETS_CACHE="../cache"
# path_dir=/your/path/to/store/model/or/dataset
# your path to model

source ./path_teamdrive.sh
path_dir=$path_team


# model setting
declare -A setting

# # base llama2-7b 4k
# setting["base"]="--model ${path_dir}/Llama-2-7b-hf/ --method pi --factor 1.0"

# # together
# setting["together"]="--model ${path_dir}/LLaMA-2-7B-32K/ --method pi --factor 8.0"

# # longlora pi 100k 
# setting["longlora"]="--model ${path_dir}/Llama-2-7b-longlora-100k-ft/ --method pi --factor 25.0"

# # codellama 100k
# setting["codellama"]="--model ${path_dir}/CodeLlama-7b-hf/ --method dy_ntk --factor 1.0 --original_max_position_embeddings 8192 --max_position_embeddings 16384"

# # yarn 64k
# setting["yarn_64k"]="--model ${path_team}/Yarn-Llama-2-7b-64k/ --method yarn --finetuned --factor 16.0"

# # yarn 128k
# setting["yarn_128k"]="--model ${path_team}/Yarn-Llama-2-7b-128k/ --method yarn --finetuned --factor 32.0"

setting["longrope_128k_mistral"]="--model ${path_team}/ft_out_model/cube-16k-mistral-128k/ck-400 --method longrope --finetuned --factor 32.0 --original_max_position_embeddings 4096 --sliding_window_attention 131072"

# longrope Mistral 256k
setting["longrope_256k_mistral"]="--model ${path_team}/ft_out_model/cube-16k-mistral-256k/ck-400 --method longrope  --finetuned --factor 64.0 --original_max_position_embeddings 4096 --sliding_window_attention 262144"

setting["longrope_mistral_128k_bf16"]="--model ${path_team}/ft_out_model/cube-mis-128k-bf16/ck-1_1000 --method longrope --finetuned --factor 32 --sliding_window_attention 131072 --original_max_position_embeddings 4096"

setting["longrope_mistral_256k_bf16_from_step500"]="--model ${path_team}/ft_out_model/cube-mis-256k-bf16-step-500/ck-1_500 --method longrope  --finetuned --sliding_window_attention 262144 --factor 64 --original_max_position_embeddings 4096"


# dataset setting
# PROOFPILE_test="--tokenized ${path_team}/proofpile-test-tokenized --dataset_min_tokens 131072 --samples 1 --truncate"

PROOFPILE_128k="--tokenized ${path_team}/proofpile-test-tokenized-mistral --dataset_min_tokens 131072 --samples 10 --truncate"
# PROOFPILE_256k="--tokenized ${path_team}/proofpile-test-tokenized --dataset_min_tokens 262144 --samples 10 --truncate"

cache_dir="../cache_dir"
output_dir=./script/ppl_eval/tmp

# save_memory="\
# --aggressive_mem_causal_lm \
# --aggressive_mem_decoder \
# --aggressive_mem_attn"
save_memory="" # check

# config_list=("base" "together" "longlora" "codellama" "yarn_64k" "yarn_128k" "longrope_128k" "longrope_256k")
# config_list=("base" "together" "longlora" "yarn_64k" "yarn_128k" "longrope_128k" "longrope_256k")

config_list=("longrope_mistral_128k_bf16" "longrope_mistral_256k_bf16_from_step500") # check

echo "dataset PROOFPILE 10sample"
# max_tokens_list=(4096 8192 32768 65536 98304 131072)
max_tokens_list=(4096 8192)
# tmp_list=(0.8 0.85 0.9 0.95 1.0 1.05 1.1 1.15 1.2 1.25 1.3)
tmp_list=(0.8 0.85 0.9 0.95 1.0 1.05 1.1 1.15 1.2 1.25 1.3 1.01 1.02 1.03 1.04 1.06 1.07 1.08 1.09 1.11 1.12 1.13 1.14)

for tmp in "${tmp_list[@]}"; do
    for config in "${config_list[@]}"; do
        for max_tokens in "${max_tokens_list[@]}"; do
            echo "####### tmp $tmp, $config, max-tokens=$max_tokens #############"
            python evaluation/perplexity.py \
                ${PROOFPILE_128k}\
                ${setting[$config]} \
                --max_tokens $max_tokens \
                --min_tokens $max_tokens \
                --tokens_step 2048 \
                --output_file "${output_dir}/proofpile_${config}_${max_tokens}_${tmp}.csv" \
                --original_max_position_embeddings 4096 \
                --flash_attn \
                ${save_memory} \
                --cache_dir $cache_dir \
                --tmps $tmp
        done
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
