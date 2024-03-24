#!/bin/bash

# ./script/ppl_eval/t2/ppl_eval_llama2_start_pi.sh 0 8192 0 ; ./script/ppl_eval/t2/ppl_eval_llama2_start_pi.sh 0 8192 2 ; ./script/ppl_eval/t2/ppl_eval_llama2_start_pi.sh 0 16384 0 ; ./script/ppl_eval/t2/ppl_eval_llama2_start_pi.sh 0 16384 2

# ./script/ppl_eval/t2/ppl_eval_llama2_start_pi.sh 1 8192 4 ; ./script/ppl_eval/t2/ppl_eval_llama2_start_pi.sh 1 8192 8 ; ./script/ppl_eval/t2/ppl_eval_llama2_start_pi.sh 1 16384 4 ; ./script/ppl_eval/t2/ppl_eval_llama2_start_pi.sh 1 16384 8

# ./script/ppl_eval/t2/ppl_eval_llama2_start_pi.sh 2 8192 16 ; ./script/ppl_eval/t2/ppl_eval_llama2_start_pi.sh 2 8192 32 ; ./script/ppl_eval/t2/ppl_eval_llama2_start_pi.sh 2 16384 16 ; ./script/ppl_eval/t2/ppl_eval_llama2_start_pi.sh 2 16384 32

# ./script/ppl_eval/t2/ppl_eval_llama2_start_pi.sh 3 8192 64 ; ./script/ppl_eval/t2/ppl_eval_llama2_start_pi.sh 3 8192 128 ; ./script/ppl_eval/t2/ppl_eval_llama2_start_pi.sh 3 16384 64 ; ./script/ppl_eval/t2/ppl_eval_llama2_start_pi.sh 3 16384 128

# 检查是否传入了足够的参数  
if [ $# -lt 3 ]; then  
    echo "Error: Insufficient arguments provided."  
    echo "Usage: $0 <CUDA_VISIBLE_DEVICES> <seq_len> <start_token>"  
    exit 1  
fi  

export CUDA_VISIBLE_DEVICES=$1

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
max_tokens=$2
start_token=$3

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
