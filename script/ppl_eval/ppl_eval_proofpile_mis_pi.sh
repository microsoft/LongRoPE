#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1
echo "gpu $1"
# export HF_DATASETS_CACHE="/path/to/store/model"
export HF_DATASETS_CACHE="../cache"
# path_dir=/your/path/to/store/model/or/dataset
# your path to model

source ./path_teamdrive.sh
path_dir=$path_team


# model setting
declare -A setting


# longrope 128k
setting["longrope_128k"]="--model ${path_dir}/ft_out_model/cube-128k-dim-piece-mono-500-#m0/ck-400/ --method longrope --finetuned --factor 32.0"

# longrope 256k
setting["longrope_256k"]="--model ${path_dir}/ft_out_model/cube_256k_from_128k/ck-600/ --method longrope --finetuned --factor 64.0"


# dataset setting
# PROOFPILE_test="--tokenized ${path_team}/proofpile-test-tokenized --dataset_min_tokens 131072 --samples 1 --truncate"
PG19_8k="--tokenized ${path_dir}/pg19-test-tokenized --samples 100 --sliding-window 256 "
PROOFPILE_128k="--tokenized ${path_team}/proofpile-test-tokenized-mistral --dataset_min_tokens 131072 --samples 10 --truncate"
PROOFPILE_256k="--tokenized ${path_team}/proofpile-test-tokenized-mistral --dataset_min_tokens 262144 --samples 10 --truncate"

cache_dir="../cache_dir"
output_dir=./script/ppl_eval

save_memory="\
--aggressive_mem_causal_lm"

# save_memory="" # check

# config_list=("base" "together" "longlora" "codellama" "yarn_64k" "yarn_128k" "longrope_128k" "longrope_256k")
# config_list=("base") # check

echo "dataset PROOFPILE 10sample"
max_tokens_list=(4096 8192 32768 65536 98304 131072)
# max_tokens_list=(4096 8192 16384 32768)

echo $path_dir

# step_list=("1_200" "1_300" "1_400" "1_500")
# step_list=()

config_list=("longrope_128k_pi") # check
setting["longrope_128k_pi"]="--model ${path_dir}/ft_out_model/cube-mis-128k-pi-re/cube-mis-128k-pi/cube-mis-128k-pi/ck-1_400/
 --method pi --factor 32.0 --sliding_window_attention 131072"

setting["longrope_128k_pi_search"]="--model ${path_dir}/ft_out_model/cube-mis-128k-pi-re/cube-mis-128k-pi/cube-mis-128k-pi/ck-1_400/
 --method pi --factor 28.8 --sliding_window_attention 131072"

max_tokens_list=(4096 8192 16384 32768 65536 98304 131072)
for config in "${config_list[@]}"; do
    for max_tokens in "${max_tokens_list[@]}"; do
        echo "####### step=$step, $config, max-tokens=$max_tokens #############"
        python evaluation/perplexity.py \
            ${PROOFPILE_128k}\
            ${setting[$config]} \
            --max_tokens $max_tokens \
            --min_tokens $max_tokens \
            --tokens_step 2048 \
            --output_file "${output_dir}/compare_pi_proofpile_${config}_${max_tokens}.csv" \
            --original_max_position_embeddings 4096 \
            --flash_attn \
            ${save_memory} \
            --cache_dir $cache_dir
    done
done

    # max_tokens_list=(262144)
    # for config in "${config_list[@]}"; do
    #     for max_tokens in "${max_tokens_list[@]}"; do
    #         echo "####### step=$step, $config, max-tokens=$max_tokens #############"
    #         python evaluation/perplexity.py \
    #             ${PROOFPILE_256k}\
    #             ${setting[$config]} \
    #             --max_tokens $max_tokens \
    #             --min_tokens $max_tokens \
    #             --tokens_step 2048 \
    #             --output_file "${output_dir}/compare_${step}_proofpile_${config}_${max_tokens}.csv" \
    #             --original_max_position_embeddings 4096 \
    #             --flash_attn \
    #             ${save_memory} \
    #             --cache_dir $cache_dir
    #     done
    # done
# done