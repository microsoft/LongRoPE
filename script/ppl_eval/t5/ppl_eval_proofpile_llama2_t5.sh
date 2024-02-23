#!/bin/bash
export CUDA_VISIBLE_DEVICES=6
# export HF_DATASETS_CACHE="/path/to/store/model"
export HF_DATASETS_CACHE="../cache"
# path_dir=/your/path/to/store/model/or/dataset
# your path to model

source ./path_teamdrive.sh
path_dir=$path_team


# model setting
declare -A setting

# base llama2-7b 4k
setting["base"]="--model ${path_dir}/Llama-2-7b-hf/ --method pi --factor 1.0"

# together
setting["together"]="--model ${path_dir}/LLaMA-2-7B-32K/ --method pi --factor 8.0"

# longlora pi 100k 
setting["longlora"]="--model ${path_dir}/Llama-2-7b-longlora-100k-ft/ --method pi --factor 25.0"

# codellama 100k
setting["codellama"]="--model ${path_dir}/CodeLlama-7b-hf/ --method dy_ntk --factor 1.0"

# yarn 64k
setting["yarn_64k"]="--model ${path_team}/Yarn-Llama-2-7b-64k --method yarn --finetuned --factor 16.0"

# yarn 128k
setting["yarn_128k"]="--model ${path_team}/Yarn-Llama-2-7b-128k --method yarn --finetuned --factor 32.0"

# longrope 128k
setting["longrope_128k"]="--model ${path_dir}/ft_out_model/cube-128k-dim-piece-mono-500-#m0/ck-400 --method s_pi --s_pi_para ./evolution/test/result_alpha/ft_s_pi_131072_result.csv --finetuned --factor 32.0"

# longrope 256k
setting["longrope_256k"]="--model ${path_dir}/ft_out_model/cube_256k_from_128k/ck-600 --method s_pi --s_pi_para ./evolution/test/result_alpha/ft_s_pi_262144_result_118.csv --finetuned --factor 64.0"


# dataset setting
PROOFPILE_128k="--tokenized ${path_team}/proofpile-test-tokenized --dataset-min-tokens 131072 --samples 10 --truncate"
PROOFPILE_256k="--tokenized ${path_team}/proofpile-test-tokenized --dataset-min-tokens 262144 --samples 10 --truncate"

cache_dir="../cache_dir"
output_dir=./evaluation/result

save_memory="\
--aggressive-mem-causal_lm \
--aggressive-mem-decoder \
--aggressive-mem-attn"
# save_memory="" # check

config_list=("base" "together" "longlora" "codellama" "yarn_64k" "yarn_128k" "longrope_128k" "longrope_256k")
config_list=("longrope_256k") # check

echo "dataset PROOFPILE 10sample"
max_tokens_list=(4096 8192 32768 65536 98304 131072)
max_tokens_list=(32768 65536 98304 131072)

for config in "${config_list[@]}"; do
    for max_tokens in "${max_tokens_list[@]}"; do
        echo "####### $config, max-tokens=$max_tokens #############"
        python evaluation/perplexity.py \
            ${PROOFPILE_128k}\
            ${setting[$config]} \
            --max-tokens $max_tokens \
            --min-tokens $max_tokens \
            --tokens-step 2048 \
            --output-file "${output_dir}/t5_proofpile_${config}_${max_tokens}.csv" \
            --original-max-position-embeddings 4096 \
            --flash_attn \
            ${save_memory} \
            --cache_dir $cache_dir
    done
done

max_tokens_list=(262144)
for config in "${config_list[@]}"; do
    for max_tokens in "${max_tokens_list[@]}"; do
        echo "####### $config, max-tokens=$max_tokens #############"
        python evaluation/perplexity.py \
            ${PROOFPILE_256k}\
            ${setting[$config]} \
            --max-tokens $max_tokens \
            --min-tokens $max_tokens \
            --tokens-step 2048 \
            --output-file "${output_dir}/t5_proofpile_${config}_${max_tokens}.csv" \
            --original-max-position-embeddings 4096 \
            --flash_attn \
            ${save_memory} \
            --cache_dir $cache_dir
    done
done
