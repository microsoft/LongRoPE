#!/bin/bash
source ./path_teamdrive.sh
path_dir=$path_team

export TOKENIZERS_PARALLELISM=false

# 检查是否传入了足够的参数  
# if [ $# -lt 1 ]; then  
#     echo "Error: Insufficient arguments provided."  
#     echo "Usage: $0 <city_idx>"  
#     exit 1  
# fi  

ck_step=1_600
prompt_name=ANTHROPIC_TEMPLATE_ORIGINAL
echo "ck_step: $ck_step"
echo "prompt_name: $prompt_name"
# ANTHROPIC_TEMPLATE_REV1


# mistral_128k="${path_team}/ft_out_model/cube-mis-128k-bf16/ck-${ck_step}"
# mistral_256k="${path_team}/ft_out_model/cube-mis-256k-bf16-step-500/ck-${ck_step}"
# mistral_256k="${path_team}/ft_out_model/longrope-256k-sft-mis/mis-256k-longalphaca-12k/ck-${ck_step}"
longrope_pose_512k="${path_team}/ft_out_model/longrope-la2-128k-pose-512k/cube-la2-128k-pose-512k/ck-1_600/"

echo "cube-la2-128k-pose-512k | needle origin"

declare -A setting

# longrope 128k
setting["longrope_pose_512k"]="-m ${longrope_pose_512k} --method longrope --finetuned --factor 128.0"
# setting["longrope_256k"]="-m ${mistral_256k} --method longrope --finetuned --factor 64.0"


# 1k, 2k, 4k, 8k, 16k, 32k, 64k, 128k, 200k, 400k, 500k, 800k, 1000k, 1500k, 1800k,  2m
# list_2m="1000,2000,4000,8000,16000,64000,128000,200000,400000,500000,800000,10000000,1500000,1800000,2000000"
# list_2m="1000,4000,64000,200000,700000,800000,1000000,1500000,1800000,2000000"
# list_2m="950000,980000,1024000,1048000,1048576"
# list_2m="1048576"
# list_2m="1024000,1048000,1550000,1850000"
# mkdir -p evaluation/needle/logs evaluation/needle/img evaluation/needle/result
list_2m="1000,2000,4000,8000,16000,64000,128000,200000,400000,500000,800000,900000"

# # clean pt
pt_list="fullmodel.pt.* gencode* cube_graph* dist_param_map.pt"
python_path=$(which python)
torch_path=$(dirname $python_path)

nums=2-0
name="${nums}-cube-la2-128k-pose-512k"
# rm -rf ./evaluation/needle/result/$name

echo "cube trace ..."
gpu_num=1

rm $pt_list
CUDA_VISIBLE_DEVICES=0 $torch_path/torchrun \
    --nproc_per_node=$gpu_num \
    --master_port 29510 \
    evaluation/needle/needle_in_haystack.py \
    --s_len 0 --e_len 256000 \
    --seq_series $list_2m \
    --city_idx 15 \
    --random_num 6072345 \
    --file_order_idx 0 \
    --use_books_idx 19 \
    --document_depth_percent_intervals 5 \
    --model_provider Llama \
    --model_path ${longrope_pose_512k} \
    --result_path ./evaluation/needle/result/$name/ \
    ${setting["longrope_pose_512k"]} \
    --flash_attn \
    --max_tokens 4000 \
    --prompt_template $prompt_name \
    --needle_type "origin" \
    --use_cube \
    --rope_method s_pi \
    --rope_tmps su \
    --use_cache \
    --tp_size $gpu_num \
    --cube_trace

# for ck in "${ck_list[@]}"; do
# 14 11 9 8 4
books_list=(19)
for books_idx in "${books_list[@]}"; do  
    # name="${nums}-cube_longrope_mis_256k_bf16_from500_ck_${ck_step}_debug_needle_origin_books_${books_idx}"
    echo "books_idx:$books_idx"

    echo "cube run ..."
    gpu_num=8
    (
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $torch_path/torchrun \
        --nproc_per_node=$gpu_num \
        --master_port 29510 \
        evaluation/needle/needle_in_haystack.py \
        --s_len 0 --e_len 2000000 \
        --seq_series $list_2m \
        --city_idx 15 \
        --random_num 6072345 \
        --file_order_idx 0 \
        --use_books_idx $books_idx \
        --document_depth_percent_intervals 5 \
        --doc_depth_series "0,12,25,38,50,62,75,88,94,100" \
        --model_provider Mistral \
        --model_path ${longrope_pose_512k} \
        --result_path ./evaluation/needle/result/$name/ \
        ${setting["longrope_pose_512k"]} \
        --flash_attn \
        --max_tokens 4000 \
        --prompt_template $prompt_name \
        --needle_type "origin" \
        --use_cube \
        --rope_method s_pi \
        --rope_tmps su \
        --use_cache \
        --tp_size $gpu_num \
        # --static_scale "2048k_mis_256k"
        
    ) 2>&1  | tee evaluation/needle/logs/eval_${name}.log

    # python evaluation/needle/visualize.py 

    python evaluation/needle/visualize.py --name $name --path evaluation/needle/result/$name/ck-$ck_step/

done