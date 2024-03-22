#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

path_dir=/your/path/to/store/model/or/dataset


model="${path_dir}/Llama-2-7b-hf/"

data_tokenized="${path_dir}/pg19_valid_mapped"
cache_dir="../cache_dir"

save_memory="\
--aggressive-mem-causal_lm \
--aggressive-mem-decoder \
--aggressive-mem-attn"
# PS: if use A6000, 48G
# seq_len > 128k use --aggressive-mem-causal_lm --aggressive-mem-decoder 
# seq_len > 256k use --aggressive-mem-causal_lm --aggressive-mem-decoder --aggressive-mem-attn


max_tokens=262144
serach_method="dim_mono"

python evolution/ppl_search_evolution.py \
    --model $model \
    --samples 5 \
    --longrope_method $serach_method \
    --longrope_init_para "./evolution/${serach_method}/init_alpha/${serach_method}_${max_tokens}.csv" \
    --factor $((max_tokens / 4096)) \
    --max-tokens $max_tokens \
    --min-tokens $max_tokens \
    --tokens-step 4000 \
    --tokenized $data_tokenized \
    --original-max-position-embeddings 4096 \
    --dataset-min-tokens $max_tokens \
    --sliding-window 65536 \
    --flash_attn \
    ${save_memory} \
    --cache_dir $cache_dir \
    --truncate 
