#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

source ./path_teamdrive.sh
path_dir=$path_team
model="${path_dir}/Llama-2-7b-hf/"

# 2*24h = 48h = 2888 min
# 8 GPUs * 4 times
# 48h / 4 = 12h


# test
PG19="--tokenized ${path_dir}/pg19-test-tokenized --samples 5 --sliding_window 256 "


cache_dir="../cache_dir"
output_dir=./evaluation/result

save_memory=""

# start_token=(0 2 4)
start_token=0 
max_tokens=8192

method_list=(dy_ntk_start)
echo "dataset PG19"
for method in "${method_list[@]}"; do
    echo "####### $method, max-tokens=$max_tokens #############"
    python evaluation/perplexity.py \
        ${PG19}\
        --model $model \
        --method "$method" \
        --factor $((max_tokens / 4096)) \
        --max_tokens $max_tokens \
        --min_tokens $max_tokens \
        --tokens_step 2048 \
        --output_file "${output_dir}/pg19_${method}_la2_${max_tokens}_start_${start_token}.csv" \
        --original_max_position_embeddings 4096 \
        --flash_attn \
        ${save_memory} \
        --cache_dir $cache_dir \
        --start_token $start_token
done

start_token=4
max_tokens=8192

for method in "${method_list[@]}"; do
    echo "####### $method, max-tokens=$max_tokens #############"
    python evaluation/perplexity.py \
        ${PG19}\
        --model $model \
        --method "$method" \
        --factor $((max_tokens / 4096)) \
        --max_tokens $max_tokens \
        --min_tokens $max_tokens \
        --tokens_step 2048 \
        --output_file "${output_dir}/pg19_${method}_la2_${max_tokens}_start_${start_token}.csv" \
        --original_max_position_embeddings 4096 \
        --flash_attn \
        ${save_memory} \
        --cache_dir $cache_dir \
        --start_token $start_token
done

start_token=0 
max_tokens=16384


for method in "${method_list[@]}"; do
    echo "####### $method, max-tokens=$max_tokens #############"
    python evaluation/perplexity.py \
        ${PG19}\
        --model $model \
        --method "$method" \
        --factor $((max_tokens / 4096)) \
        --max_tokens $max_tokens \
        --min_tokens $max_tokens \
        --tokens_step 2048 \
        --output_file "${output_dir}/pg19_${method}_la2_${max_tokens}_start_${start_token}.csv" \
        --original_max_position_embeddings 4096 \
        --flash_attn \
        ${save_memory} \
        --cache_dir $cache_dir \
        --start_token $start_token
done

start_token=4
max_tokens=16384


for method in "${method_list[@]}"; do
    echo "####### $method, max-tokens=$max_tokens #############"
    python evaluation/perplexity.py \
        ${PG19}\
        --model $model \
        --method "$method" \
        --factor $((max_tokens / 4096)) \
        --max_tokens $max_tokens \
        --min_tokens $max_tokens \
        --tokens_step 2048 \
        --output_file "${output_dir}/pg19_${method}_la2_${max_tokens}_start_${start_token}.csv" \
        --original_max_position_embeddings 4096 \
        --flash_attn \
        ${save_memory} \
        --cache_dir $cache_dir \
        --start_token $start_token
done
