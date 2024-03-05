#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
source ./path_teamdrive.sh
path_dir=$path_team
export HF_DATASETS_CACHE="../cache"
# longrope 128k
setting["longrope_128k"]="--model ${path_dir}/ft_out_model/cube-128k-dim-piece-mono-500-#m0/ck-400/ --method longrope --finetuned --factor 32.0"

# longrope 256k
setting["longrope_256k"]="--model ${path_dir}/ft_out_model/cube_256k_from_128k/ck-600/ --method longrope --finetuned --factor 64.0"
cache_dir="../cache_dir"
output_dir=./evaluation/result

save_memory=""

python evaluation/prompt-loop.py \
    ${setting["longrope_256k"]} \
    --max_tokens 4096 \
