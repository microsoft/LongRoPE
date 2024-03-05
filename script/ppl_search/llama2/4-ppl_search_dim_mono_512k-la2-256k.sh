#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

path_dir=/your/path/to/store/model/or/dataset

source ./path_teamdrive.sh
path_dir=$path_team
# model=${path_dir}/longrope-llama2-256k/
model=${path_team}/ft_out_model/cube_256k_from_128k/ck-600
model_ft_len=262144

# dataset
BOOK3_VALID_MISTRAL="--tokenized ${path_dir}/books3-valid-sampled-1024k-tokenized --samples 3 --truncate"
PG19_8k="--tokenized ${path_dir}/pg19-test-tokenized --samples 1 --truncate"

cache_dir="../cache_dir"

max_tokens=524288
serach_method="dim_mono"

save_memory="\
--aggressive_mem_causal_lm \
--aggressive_mem_decoder \
--aggressive_mem_attn"

save_memory="\
--aggressive_mem_causal_lm \
--aggressive_mem_decoder"

python evolution/ppl_search_evolution.py \
    ${BOOK3_VALID_MISTRAL} \
    --model $model \
    --longrope_method $serach_method \
    --search_twice \
    --finetuned \
    --longrope_init_para "./evolution/${serach_method}/init_alpha/${serach_method}_yarn_$((max_tokens / model_ft_len * 4096 )).csv" \
    --factor $((max_tokens / model_ft_len)) \
    --max_tokens $max_tokens \
    --min_tokens $max_tokens \
    --tokens_step 4000 \
    --dataset_min_tokens $max_tokens \
    --original_max_position_embeddings 4096 \
    --flash_attn \
    ${save_memory} \
    --cache_dir $cache_dir 

