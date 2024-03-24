#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
# export HF_DATASETS_CACHE="/path/to/store/model"
export HF_DATASETS_CACHE="../cache"
# path_dir=/your/path/to/store/model/or/dataset
# your path to model

# source ./path_teamdrive.sh
# path_dir=$path_team
TEAMDRIVE="/mnt/cz"
params=$TEAMDRIVE/models/longrope_params/phi-2.5_v3/init/init_params_dm_128k_dy_8k.csv
model_phi=$TEAMDRIVE/models/phi-25_v3_longrope_128k_redpajama_dpm_32_400
tokenized=$TEAMDRIVE/datasets/phi2.5/proofpile-test-tokenized

# model setting
declare -A setting

# longrope 256k
setting["longrope_256k"]="--model ${path_dir}/ft_out_model/cube_256k_from_128k/ck-600/ --method longrope --finetuned --factor 64.0"

# phi_2.5
setting["phi_2_5"]="--model $model_phi --method longrope --longrope_para $params --factor 64.0"

# dataset setting
# PROOFPILE_test="--tokenized ${path_team}/proofpile-test-tokenized --dataset_min_tokens 131072 --samples 1 --truncate"

PROOFPILE_128k="--tokenized $tokenized --dataset_min_tokens 131072 --samples 10 --truncate"


cache_dir="../cache_dir"
output_dir=./evaluation/result

# save_memory="\
# --aggressive_mem_causal_lm \
# --aggressive_mem_decoder \
# --aggressive_mem_attn"
save_memory="--aggressive_mem_causal_lm" # check

config_list=("base" "together" "longlora" "codellama" "yarn_64k" "yarn_128k" "longrope_128k" "longrope_256k")
# config_list=("base" "together" "longlora" "yarn_64k" "yarn_128k" "longrope_128k" "longrope_256k")

config_list=("phi_2_5") # check

echo "dataset PROOFPILE 10sample"
max_tokens_list=(4096 8192 32768 65536 98304 131072)
max_tokens_list=(4096)

for config in "${config_list[@]}"; do
    for max_tokens in "${max_tokens_list[@]}"; do
        echo "####### $config, max-tokens=$max_tokens #############"
        python evaluation/perplexity.py \
            ${PROOFPILE_128k}\
            ${setting[$config]} \
            --max_tokens $max_tokens \
            --min_tokens $max_tokens \
            --tokens_step 2048 \
            --output_file "${output_dir}/t5_proofpile_${config}_${max_tokens}.csv" \
            --original_max_position_embeddings 4096 \
            --flash_attn \
            ${save_memory} \
            --cache_dir $cache_dir
    done
done
