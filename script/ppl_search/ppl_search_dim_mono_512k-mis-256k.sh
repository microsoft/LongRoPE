#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

path_dir=/your/path/to/store/model/or/dataset

model=${path_dir}/ft_mis_256k/
model_ft_len=262144

# dataset
BOOK3_VALID_MISTRAL="--tokenized ${path_dir}/books3-valid-sampled-1024k-tokenized-mistral --samples 3 --truncate"
cache_dir="../cache_dir"

max_tokens=524288
serach_method="dim_mono"

save_memory="\
--aggressive-mem-causal_lm \
--aggressive-mem-decoder \
--aggressive-mem-attn"
# PS: if use A6000, 48G
# seq_len > 128k use --aggressive-mem-causal_lm --aggressive-mem-decoder 
# seq_len > 256k use --aggressive-mem-causal_lm --aggressive-mem-decoder --aggressive-mem-attn


python evolution/ppl_search_evolution.py \
    ${BOOK3_VALID_MISTRAL} \
    --model $model \
    --s_pi_method $serach_method \
    --search_twice \
    --finetuned \
    --s_pi_init_para "./evolution/${serach_method}/init_alpha/${serach_method}_$((max_tokens / model_ft_len * 4096 )).csv" \
    --factor $((max_tokens / model_ft_len)) \
    --max-tokens $max_tokens \
    --min-tokens $max_tokens \
    --tokens-step 4000 \
    --dataset-min-tokens $max_tokens \
    --original-max-position-embeddings 4096 \
    --flash_attn \
    ${save_memory} \
    --cache_dir $cache_dir

