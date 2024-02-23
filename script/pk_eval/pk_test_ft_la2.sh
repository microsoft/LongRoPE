#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

path_dir=/your/path/to/store/model/or/dataset
# ${path_dir}

cache_dir="../cache_dir"

declare -A setting

# spi 128k
setting["s_pi_la2_ft"]="--model ${path_dir}/ft_la2_256k/ --finetuned --method s_pi --factor 64.0"
setting["s_pi_mis_ft"]="--model ${path_dir}/ft_mis_256k/ --finetuned --method s_pi --factor 64.0"


tokens_list=(262144)
# method_list=(pi_ft dy_ntk_ft yarn_ft dy_yarn_ft s_pi_ft dy_s_pi_ft)
method_list=(s_pi_la2_ft)
for method in "${method_list[@]}"; do
    for len_tokens in "${tokens_list[@]}"; do
        echo "############################################################"
        echo "############################################################"
        echo "####### method $method, max-tokens=$len_tokens #############"
        python evaluation/passkey.py \
            ${setting[$method]} \
            --max-tokens $len_tokens \
            --min-tokens $len_tokens \
            --tokens-step 2048 \
            --length-step 1024 \
            --iterations 10 \
            --flash_attn \
            --cache_dir "../cache_dir" \
            --original-max-position-embeddings 4096 \
            --output-file "./script/pk_eval/${method}_${len_tokens}_pk_itr10.csv"
    done
done
