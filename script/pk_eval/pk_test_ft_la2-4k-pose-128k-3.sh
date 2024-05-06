#!/bin/bash
export CUDA_VISIBLE_DEVICES=2

path_dir=/your/path/to/store/model/or/dataset
# ${path_dir}

cache_dir="../cache_dir"

declare -A setting
source ./path_teamdrive.sh
path_dir=$path_team

# spi 128k
# setting["s_pi_la2_ft"]="--model ${path_dir}/ft_la2_256k/ --finetuned --method s_pi --factor 64.0"
# setting["longrope_128k"]="--model ${path_dir}/ft_out_model/cube-128k-dim-piece-mono-500-#m0/ck-400/ --method longrope --finetuned --factor 32.0"

# setting["s_pi_mis_ft"]="--model ${path_dir}/ft_mis_256k/ --finetuned --method s_pi --factor 64.0"

# setting["longrope_128k_pose_4k"]="--model ${path_dir}/ft_out_model/cube-la2-4k-128k/ck-7_1000/ --method longrope --finetuned --factor 32.0"



#  longrope 128k pose 4k
setting["longrope_128k_pose_4k-fliter-step1000"]="--model ${path_dir}/ft_out_model/cube-la2-4k-128k-same-doc/cube-la2-4k-128k-same-doc/checkpoint/ck-2_1000 --method longrope --finetuned --factor 32.0"
# /mnt/yiran/teamdrive3/ExtendSeqLen/ft_out_model/cube-la2-4k-128k-same-doc/cube-la2-4k-128k-same-doc/checkpoint/ck-2_1000

setting["longrope_128k_pose_4k-fliter-step600"]="--model ${path_dir}/ft_out_model/cube-la2-4k-128k-same-doc/cube-la2-4k-128k-same-doc/checkpoint/ck-1_600 --method longrope --finetuned --factor 32.0"


setting["longrope_128k_pose_4k-pad-step1000"]="--model ${path_dir}/ft_out_model/cube-la2-4k-128k-same-doc-pad/cube-la2-4k-128k-same-doc-pad/checkpoint/ck-1_1000 --method longrope --finetuned --factor 32.0"
# /mnt/yiran/teamdrive3/ExtendSeqLen/ft_out_model/cube-la2-4k-128k-same-doc-pad/cube-la2-4k-128k-same-doc-pad/checkpoint/ck-1_1000

setting["longrope_128k_pose_4k-pad-step600"]="--model ${path_dir}/ft_out_model/cube-la2-4k-128k-same-doc-pad/cube-la2-4k-128k-same-doc-pad/checkpoint/ck-1_1000 --method longrope --finetuned --factor 32.0"

# tokens_list=(16384 32768 65536 102400 
# tokens_list=(131072 163840)
tokens_list=(131072 163840 262144)

# method_list=(pi_ft dy_ntk_ft yarn_ft dy_yarn_ft s_pi_ft dy_s_pi_ft)
method_list=( \
    # "longrope_128k_pose_4k-fliter-step1000" \
    # "longrope_128k_pose_4k-fliter-step600" \
    "longrope_128k_pose_4k-pad-step1000" \
    # "longrope_128k_pose_4k-pad-step600" \
    ) # check
for method in "${method_list[@]}"; do
    for len_tokens in "${tokens_list[@]}"; do
        echo "############################################################"
        echo "############################################################"
        echo "####### method $method, max-tokens=$len_tokens #############"
        python evaluation/passkey.py \
            ${setting[$method]} \
            --max-tokens $len_tokens \
            --min-tokens $len_tokens \
            --tokens-step 2048 \
            --length-step 1024 \
            --iterations 10 \
            --flash_attn \
            --cache_dir "../cache_dir" \
            --original_max_position_embeddings 4096 \
            --output-file "./script/pk_eval/${method}_${len_tokens}_pk_itr10.csv" \
            --aggressive_mem_decoder \
            --aggressive_mem_causal_lm \
            --aggressive_mem_attn
    done
done
