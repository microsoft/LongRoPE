# path_dir=/your/path/to/store/model/or/dataset
# your path to model

source ./path_teamdrive.sh
path_dir=$path_team


# model setting
declare -A setting

# longlora pi 100k 
setting["longlora"]="--model ${path_dir}/Llama-2-7b-longlora-100k-ft/ --method pi --factor 25.0"

# codellama 100k
setting["codellama"]="--model ${path_dir}/CodeLlama-7b-hf/ --method dy_ntk --factor 1.0"

# longrope 128k
setting["longrope_128k"]="--model ${path_dir}/ft_out_model/cube-128k-dim-piece-mono-500-#m0/ck-400 --method longrope --longrope_para ./evolution/test/result_alpha/ft_longrope_131072_result.csv --finetuned --factor 32.0"

# longrope 256k
setting["longrope_256k"]="--model ${path_dir}/ft_out_model/cube_256k_from_128k/ck-600 --method longrope --longrope_para ./evolution/test/result_alpha/ft_longrope_262144_result_118.csv --finetuned --factor 64.0"


# dataset setting
PG19_8k="--tokenized ${path_dir}/pg19-test-tokenized --samples 100 --sliding_window 256 "
PG19_long="--tokenized ${path_dir}/pg19-test-tokenized --samples 100 --sliding_window 4096 "


cache_dir="../cache_dir"
output_dir=./evaluation/result

save_memory="\
--aggressive-mem-causal_lm \
--aggressive-mem-decoder \
--aggressive-mem-attn"
save_memory="" # check

config_list=("longlora" "codellama" "longrope_128k" "longrope_256k")
config_list=("longrope_128k") # check

echo "dataset PG19 100sample"
max_tokens=16384
gpu_num=1
for config in "${config_list[@]}"; do
    echo "####### $config, max-tokens=$max_tokens #############"
    CUDA_VISIBLE_DEVICES=0,1,2,3 /mnt/yiran/miniconda3/envs/cube4infer/bin/torchrun \
        --nproc_per_node=$gpu_num \
        --master_port 29510 \
        evaluation/perplexity.py \
        ${PG19_8k}\
        ${setting[$config]} \
        --max_tokens $max_tokens \
        --min_tokens $max_tokens \
        --tokens_step 2048 \
        --output_file "${output_dir}/t7_pg19_${config}_${max_tokens}.csv" \
        --original_max_position_embeddings 4096 \
        --flash_attn \
        ${save_memory} \
        --cache_dir $cache_dir \
        --use_cube \
        --rope_method s_pi \
        --rope_tmps su \
        --use_cache \
        --tp_size $gpu_num \
        --cube_trace
done

max_tokens=16384
gpu_num=4
for config in "${config_list[@]}"; do
    echo "####### $config, max-tokens=$max_tokens #############"
    CUDA_VISIBLE_DEVICES=0,1,2,3 /mnt/yiran/miniconda3/envs/cube4infer/bin/torchrun \
        --nproc_per_node=$gpu_num \
        --master_port 29510 \
        evaluation/perplexity.py \
        ${PG19_8k}\
        ${setting[$config]} \
        --max_tokens $max_tokens \
        --min_tokens $max_tokens \
        --tokens_step 2048 \
        --output_file "${output_dir}/t7_pg19_${config}_${max_tokens}.csv" \
        --original_max_position_embeddings 4096 \
        --flash_attn \
        ${save_memory} \
        --cache_dir $cache_dir \
        --use_cube \
        --rope_method s_pi \
        --rope_tmps su \
        --use_cache \
        --tp_size $gpu_num
done

# max_tokens=65535
# for config in "${config_list[@]}"; do
#     echo "####### $config, max-tokens=$max_tokens #############"
#     python evaluation/perplexity.py \
#         ${PG19_long}\
#         ${setting[$config]} \
#         --max-tokens $max_tokens \
#         --min-tokens $max_tokens \
#         --tokens-step 2048 \
#         --output-file "${output_dir}/t7_pg19_${config}_${max_tokens}.csv" \
#         --original-max-position-embeddings 4096 \
#         --flash_attn \
#         ${save_memory} \
#         --cache_dir $cache_dir
# done

# max_tokens=131072
# for config in "${config_list[@]}"; do
#     echo "####### $config, max-tokens=$max_tokens #############"
#     python evaluation/perplexity.py \
#         ${PG19_long}\
#         ${setting[$config]} \
#         --max-tokens $max_tokens \
#         --min-tokens $max_tokens \
#         --tokens-step 2048 \
#         --output-file "${output_dir}/t7_pg19_${config}_${max_tokens}.csv" \
#         --original-max-position-embeddings 4096 \
#         --flash_attn \
#         ${save_memory} \
#         --cache_dir $cache_dir
# done