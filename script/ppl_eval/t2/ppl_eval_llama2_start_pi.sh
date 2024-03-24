#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

source ./path_teamdrive.sh
path_dir=$path_team
model="${path_dir}/Llama-2-7b-hf/"

# 2*24h = 48h = 2888 min
# 8 GPUs * 4 times
# 48h / 4 = 12h

# sliding win = 256, 
# pg19 5sample
#   8k 0.5h 16k 1h
# pg19 100sample
#   8k 10h 16k 20h avg 15h


# test
PG19="--tokenized ${path_dir}/pg19-test-tokenized --samples 100 --sliding_window 4096 "


cache_dir="../cache_dir"
output_dir=./evaluation/result

save_memory=""

# start_token=(0 2 4)
start_token=0 
max_tokens=8192

method_list=(pi_start)
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

method_list=(pi_start)
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

start_token=0 
max_tokens=16384

method_list=(pi_start)
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
max_tokens=16384

method_list=(pi_start)
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
