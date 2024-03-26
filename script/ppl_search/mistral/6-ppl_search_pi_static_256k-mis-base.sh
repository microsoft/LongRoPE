#!/bin/bash
export CUDA_VISIBLE_DEVICES=7

path_dir=/your/path/to/store/model/or/dataset

source ./path_teamdrive.sh
path_dir=$path_team
model="${path_dir}/Mistral-7B-v0.1"

data_tokenized="${path_dir}/pg19-valid-tokenized-mistral/"
cache_dir="../cache_dir"

save_memory="\
--aggressive_mem_causal_lm \
--aggressive_mem_decoder"


# max_tokens=262144
max_tokens=262144
serach_method="pi_static"
scale=64.0

mkdir -p evolution/log/$serach_method

python evolution/ppl_search_evolution.py \
    --model $model \
    --samples 1 \
    --longrope_method $serach_method \
    --longrope_init_para ./evolution/pi_static/pi_static_1x.csv \
    --factor $((max_tokens / 4096)) \
    --max_tokens $max_tokens \
    --min_tokens $max_tokens \
    --tokens_step 4000 \
    --tokenized $data_tokenized \
    --original_max_position_embeddings 4096 \
    --dataset_min_tokens $max_tokens \
    --sliding_window_attention $max_tokens \
    --flash_attn \
    ${save_memory} \
    --cache_dir $cache_dir \
    --truncate 
