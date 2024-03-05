#!/bin/bash
export CUDA_VISIBLE_DEVICES=7

# path_dir=/your/path/to/store/model/or/dataset
# your path to model

source ./path_teamdrive.sh
path_dir=$path_team
model="${path_dir}/Llama-2-7b-hf/"


# test
PG19_t="--tokenized ${path_dir}/pg19-test-tokenized --samples 100 --sliding-window 256 "
PG19="--tokenized ${path_dir}/pg19-test-tokenized --samples 100 --sliding-window 256 "
PROOFPILE_LONG_SMALL="--tokenized ${path_dir}/proofpile-test-tokenized --dataset-min-tokens 131072 --samples 10 --truncate"
BOOK3="--tokenized ${path_dir}/books3-test-sampled-1024k-tokenized --samples 20 --sliding-window 262144 --dataset-min-tokens 2097152"


cache_dir="../cache_dir"
output_dir=./evaluation/result

save_memory="\
--aggressive-mem-causal_lm \
--aggressive-mem-decoder \
--aggressive-mem-attn"


max_tokens=4100
# method_list=(pi dy_ntk yarn s_pi)
method_list=(longlora)
echo "dataset PG19"
for method in "${method_list[@]}"; do
    echo "####### $method, max-tokens=$max_tokens #############"
    python evaluation/perplexity.py \
        ${PG19}\
        --model $model \
        --method "$method" \
        --factor $((max_tokens / 4096)) \
        --longrope_para "./evolution/search_result/final-dim_mono-4100-it-4_1.csv" \
        --max_tokens $max_tokens \
        --min_tokens $max_tokens \
        --tokens_step 2048 \
        --original_max_position_embeddings 4096 \
        --flash_attn \
        ${save_memory} \
        --cache_dir $cache_dir
done
