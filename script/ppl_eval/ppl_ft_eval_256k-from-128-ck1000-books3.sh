#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

source ./path_teamdrive.sh
# ${path_team}
# path_team=/app/cache_dir
# model="${path_team}/Llama-2-7b-hf/"

# model_lora_pi="${path_team}/ft_out_model/checkpoint-1000-pi-8k"

ck_step=1000
# model_lora_s_pi="${path_team}/ft_out_model/checkpoint-1000-s-pi-8k-lora-full"
# model_ft_s_pi_ck100="${path_team}/ft_cube_128k/ck-100/"
# /home/aisilicon/teamdrive/ExtendSeqLen/ft_cube_256k-2/ck-1

model_ft_s_pi="${path_team}/ft_out_model/cube_256k_from_128k/ck-${ck_step}/"
# /mnt/yiran/teamdrive/ExtendSeqLen/ft_out_model/cube_256k_dim_mono/ck-500
# model_ft_s_pi="${path_team}/ft_out_model/cube_256k_from_128k/ck-100"
# model_ft_s_pi=/app/cache_dir/ck-150/
# model_ft_s_pi="${path_team}/ft_cube_128k/ck-${ck_step}/"
# /data/yiran/teamdrive2/ExtendSeqLen/ft_cube_128k/ck-700

echo "model:${model_ft_s_pi}"
# model_lora_yarn="${path_team}/ft_out_model/checkpoint-1000-yarn-8k-lora-full-nonshift"
# model_ft_yarn="${path_team}/ft_out_model/checkpoint-1000-yarn-8k-ft-full-4"


# proof-pile test
PROOFPILE_LONG_SMALL="--tokenized ${path_team}/proofpile-test-tokenized --dataset-min-tokens 262144 --samples 10 --truncate"
# PG19="--tokenized ${path_team}/pg19-test-tokenized --dataset-min-tokens 16384 --samples 5"
# BOOK3="--tokenized ${path_team}/books3-test-sampled-1024k-tokenized --dataset-min-tokens 2097152 --samples 20 --sliding-window 262144"

cache_dir="../cache_dir"

# 定义关联数组
declare -A setting

setting["s_pi"]="--model $model --method s_pi"
# setting["s_pi_lora"]="--model $model --peft-model $model_lora_s_pi --method s_pi"
setting["s_pi_ft"]="--model $model_ft_s_pi --method s_pi --finetuned "

token_list=(8192 16384 32768)
method_list=(s_pi_ft)
for method in "${method_list[@]}"; do
    for max_tokens in "${token_list[@]}"; do
        # max_tokens=65536
        echo "####### $method, max-tokens=$max_tokens #############"
        python evaluation/perplexity.py \
            ${BOOK3}\
            --model ${path_team}/Llama-2-7b-longlora-100k-ft/ \
            --method "$method" \
            --factor 25.0 \
            --max-tokens $max_tokens \
            --min-tokens $max_tokens \
            --tokens-step 2048 \
            --output-file "./evaluation/result/step1000-book3_20sample_${method}_la2_${max_tokens}.csv" \
            --original-max-position-embeddings 4096 \
            --flash_attn \
            --aggressive-mem-causal_lm \
            --aggressive-mem-decoder \
            --cache_dir $cache_dir
    done
done