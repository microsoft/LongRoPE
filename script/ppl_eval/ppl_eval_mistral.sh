#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

path_dir=/your/path/to/store/model/or/dataset

# your path to model

model="${path_dir}/Llama-2-7b-hf/"


# test
PG19="--tokenized ${path_dir}/pg19-test-tokenized-mistral --samples 100 --sliding-window 256 "
PROOFPILE_LONG_SMALL="--tokenized ${path_dir}/proofpile-test-tokenized --dataset-min-tokens 131072 --samples 10 --truncate"
BOOK3="--tokenized ${path_dir}/books3-test-sampled-1024k-tokenized-mistral --samples 20 --sliding-window 262144 --dataset-min-tokens 2097152"


cache_dir="../cache_dir"
output_dir=./evaluation/result

save_memory="\
--aggressive-mem-causal_lm \
--aggressive-mem-decoder \
--aggressive-mem-attn"
# PS: if use A6000, 48G
# seq_len > 128k use --aggressive-mem-causal_lm --aggressive-mem-decoder 
# seq_len > 256k use --aggressive-mem-causal_lm --aggressive-mem-decoder --aggressive-mem-attn


max_tokens=8192
# method_list=(pi dy_ntk yarn s_pi)
method_list=(pi)
echo "dataset PG19"
for method in "${method_list[@]}"; do
    echo "####### $method, max-tokens=$max_tokens #############"
    python evaluation/perplexity.py \
        ${PG19}\
        --model $model \
        --method "$method" \
        --factor $((max_tokens / 4096)) \
        --s_pi_para "./evolution/dim_mono/result_alpha/dim_mono_8192_result.csv" \
        --max-tokens $max_tokens \
        --min-tokens $max_tokens \
        --tokens-step 2048 \
        --output-file "${output_dir}/pg19_${method}_la2_${max_tokens}.csv" \
        --original-max-position-embeddings 4096 \
        --flash_attn \
        ${save_memory} \
        --cache_dir $cache_dir
done

echo "dataset PROOFPILE_LONG_SMALL"
for method in "${method_list[@]}"; do
    echo "####### $method, max-tokens=$max_tokens #############"
    python evaluation/perplexity.py \
        ${PROOFPILE_LONG_SMALL}\
        --model $model \
        --method "$method" \
        --factor $((max_tokens / 4096)) \
        --s_pi_para "./evolution/dim_mono/result_alpha/dim_mono_8192_result.csv" \
        --max-tokens $max_tokens \
        --min-tokens $max_tokens \
        --tokens-step 2048 \
        --output-file "${output_dir}/proofpile_${method}_la2_${max_tokens}.csv" \
        --original-max-position-embeddings 4096 \
        --flash_attn \
        ${save_memory} \
        --cache_dir $cache_dir
done


echo "dataset books3"
max_tokens=2097152
method_list=(s_pi)

model=${path_dir}/ft_mis_256k/

echo "dataset BOOK3"
for method in "${method_list[@]}"; do
    echo "####### $method, max-tokens=$max_tokens #############"
    python evaluation/perplexity.py \
        ${PG19}\
        --model $model \
        --finetuned \
        --method "$method" \
        --factor $((max_tokens / 131072)) \
        --max-tokens $max_tokens \
        --min-tokens $max_tokens \
        --tokens-step 2048 \
        --output-file "${output_dir}/proofpile_${method}_la2_${max_tokens}.csv" \
        --original-max-position-embeddings 4096 \
        --sliding_window_attention $max_tokens \
        --flash_attn \
        ${save_memory} \
        --cache_dir $cache_dir
done
