#!/bin/bash
export CUDA_VISIBLE_DEVICES=3

source ./path_teamdrive.sh
path_dir=$path_team

# model="${path_team}/Llama-2-7b-hf/"

# model_128="${path_dir}/ft_out_model/cube-128k-dim-piece-mono-500-#m0/ck-400/"
# model_256="${path_dir}/ft_out_model/cube_256k_from_128k/ck-600/"
# cube-mistral-128k

mistral_128k="${path_team}/ft_out_model/cube-16k-mistral-128k/ck-400"
mistral_256k="${path_team}/ft_out_model/cube-16k-mistral-256k/ck-400"

declare -A setting

# longrope 128k
setting["longrope_128k"]="-m ${mistral_128k} --method longrope --longrope_para /mnt/yiran/2048k-mistral-256k/s-PI/evolution/test/result_alpha/mistral_131072_dim_mono_ppl12.csv --factor 32.0 --sliding_window_attention 131072"

# longrope 256k
setting["longrope_256k"]="-m ${mistral_256k} --method longrope --longrope_para /mnt/yiran/2048k-mistral-256k/s-PI/evolution/test/result_alpha/mistral_262144_dim_mono_ppl9.csv --factor 64.0 --sliding_window_attention 262144"


# mkdir -p evaluation/needle/logs evaluation/needle/img evaluation/needle/result

(
python -u evaluation/needle/needle_in_haystack.py \
    --s_len 0 --e_len 128000 \
    --context_lengths_min 1024 \
    --context_lengths_max 128000 \
    --context_lengths_num_intervals 40 \
    --model_provider Mistral \
    --model_path ${mistral_256k} \
    --result_path ./evaluation/needle/result/longrope_mis_256k/ \
    ${setting["longrope_256k"]} \
    --flash_attn \
    --max_tokens 4000 \

) 2>&1  | tee evaluation/needle/logs/eval_longrope_mis_256k.log

# python evaluation/needle/visualize.py 
