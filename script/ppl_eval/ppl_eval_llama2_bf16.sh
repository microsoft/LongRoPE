#!/bin/bash
export CUDA_VISIBLE_DEVICES=3

# path_dir=/your/path/to/store/model/or/dataset
# your path to model

source ./path_teamdrive.sh
path_dir=$path_team
# model="${path_dir}/Llama-2-7b-hf/"
model="${path_dir}/Llama-2-7B-bf16-sharded"

# test
# PG19="--tokenized ${path_dir}/pg19-test-tokenized --samples 100 --sliding-window 256 "
PROOFPILE_LONG_SMALL="--tokenized ${path_dir}/proofpile-test-tokenized --dataset_min_tokens 131072 --samples 10 --truncate"
# BOOK3="--tokenized ${path_dir}/books3-test-sampled-1024k-tokenized --samples 20 --sliding-window 262144 --dataset-min-tokens 2097152"


cache_dir="../cache_dir"
output_dir=./evaluation/result

save_memory="\
--aggressive_mem_causal_lm"
# PS: if use A6000, 48G
# seq_len > 128k use --aggressive-mem-causal_lm --aggressive-mem-decoder 
# seq_len > 256k use --aggressive-mem-causal_lm --aggressive-mem-decoder --aggressive-mem-attn


echo "dataset PROOFPILE 10sample"
# max_tokens_list=(4096 8192 32768 65536 98304 131072)
max_tokens_list=(4096 8192 )

echo $path_dir


PROOFPILE_128k="--tokenized ${path_team}/proofpile-test-tokenized --dataset_min_tokens 131072 --samples 10 --truncate"

# step_list=("1_200" "1_300" "1_400" "1_500")
step_list=("400")

config_list=("longrope_la2_128k_bf16") # check
# setting["longrope_la2_128k_fp16"]="--model ${path_dir}/Llama-2-7b-hf --method pi --factor 1.0"
setting["longrope_la2_128k_bf16"]="--model ${path_dir}/Llama-2-7B-bf16-sharded --method pi --factor 1.0"

for step in "${step_list[@]}"; do
    # setting["longrope_128k_bf16"]="--model ${path_dir}/ft_out_model/cube-mis-128k-bf16/ck-${step}/ --method longrope --finetuned --factor 32.0 --sliding_window_attention 131072"
    max_tokens_list=(4096)
    for config in "${config_list[@]}"; do
        for max_tokens in "${max_tokens_list[@]}"; do
            echo "####### step=$step, $config, max-tokens=$max_tokens #############"
            python evaluation/perplexity.py \
                ${PROOFPILE_128k}\
                ${setting[$config]} \
                --max_tokens $max_tokens \
                --min_tokens $max_tokens \
                --tokens_step 2048 \
                --output_file "${output_dir}/compare_fp16_proofpile_${config}_${max_tokens}.csv" \
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
done