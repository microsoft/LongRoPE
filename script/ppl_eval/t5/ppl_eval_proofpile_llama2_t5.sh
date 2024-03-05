#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
# export HF_DATASETS_CACHE="/path/to/store/model"
export HF_DATASETS_CACHE="../cache"
# path_dir=/your/path/to/store/model/or/dataset
# your path to model

source ./path_teamdrive.sh
path_dir=$path_team


# model setting
declare -A setting

# base llama2-7b 4k
setting["base"]="--model ${path_dir}/Llama-2-7b-hf/ --method pi --factor 1.0"

# together
setting["together"]="--model ${path_dir}/LLaMA-2-7B-32K/ --method pi --factor 8.0"

# longlora pi 100k 
setting["longlora"]="--model ${path_dir}/Llama-2-7b-longlora-100k-ft/ --method pi --factor 25.0"

# codellama 100k
setting["codellama"]="--model ${path_dir}/CodeLlama-7b-hf/ --method dy_ntk --factor 1.0 --original_max_position_embeddings 8192 --max_position_embeddings 16384"

# yarn 64k
setting["yarn_64k"]="--model ${path_team}/Yarn-Llama-2-7b-64k/ --method yarn --finetuned --factor 16.0"

# yarn 128k
setting["yarn_128k"]="--model ${path_team}/Yarn-Llama-2-7b-128k/ --method yarn --finetuned --factor 32.0"

# longrope 128k
setting["longrope_128k"]="--model ${path_dir}/ft_out_model/cube-128k-dim-piece-mono-500-#m0/ck-400/ --method longrope --finetuned --factor 32.0"

# longrope 256k
setting["longrope_256k"]="--model ${path_dir}/ft_out_model/cube_256k_from_128k/ck-600/ --method longrope --finetuned --factor 64.0"


# dataset setting
# PROOFPILE_test="--tokenized ${path_team}/proofpile-test-tokenized --dataset_min_tokens 131072 --samples 1 --truncate"

PROOFPILE_128k="--tokenized ${path_team}/proofpile-test-tokenized --dataset_min_tokens 131072 --samples 10 --truncate"
PROOFPILE_256k="--tokenized ${path_team}/proofpile-test-tokenized --dataset_min_tokens 262144 --samples 10 --truncate"

cache_dir="../cache_dir"
output_dir=./evaluation/result

# save_memory="\
# --aggressive_mem_causal_lm \
# --aggressive_mem_decoder \
# --aggressive_mem_attn"
save_memory="--aggressive_mem_causal_lm" # check

config_list=("base" "together" "longlora" "codellama" "yarn_64k" "yarn_128k" "longrope_128k" "longrope_256k")
# config_list=("base" "together" "longlora" "yarn_64k" "yarn_128k" "longrope_128k" "longrope_256k")

config_list=("codellama") # check

echo "dataset PROOFPILE 10sample"
max_tokens_list=(4096 8192 32768 65536 98304 131072)
# max_tokens_list=(4096)

# for config in "${config_list[@]}"; do
#     for max_tokens in "${max_tokens_list[@]}"; do
#         echo "####### $config, max-tokens=$max_tokens #############"
#         python evaluation/perplexity.py \
#             ${PROOFPILE_128k}\
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

config_list=("codellama")
max_tokens_list=(4096 131072)
for config in "${config_list[@]}"; do
    for max_tokens in "${max_tokens_list[@]}"; do
        echo "####### $config, max-tokens=$max_tokens #############"
        python evaluation/perplexity.py \
            ${PROOFPILE_128k}\
            ${setting[$config]} \
            --max_tokens $max_tokens \
            --min_tokens $max_tokens \
            --tokens_step 2048 \
            --output_file "${output_dir}/t5_proofpile_8k_${config}_${max_tokens}.csv" \
            --flash_attn \
            ${save_memory} \
            --cache_dir $cache_dir
    done
done

max_tokens_list=(262144)
for config in "${config_list[@]}"; do
    for max_tokens in "${max_tokens_list[@]}"; do
        echo "####### $config, max-tokens=$max_tokens #############"
        python evaluation/perplexity.py \
            ${PROOFPILE_256k}\
            ${setting[$config]} \
            --max_tokens $max_tokens \
            --min_tokens $max_tokens \
            --tokens_step 2048 \
            --output_file "${output_dir}/t5_proofpile_8k_${config}_${max_tokens}.csv" \
            --flash_attn \
            ${save_memory} \
            --cache_dir $cache_dir
    done
done
