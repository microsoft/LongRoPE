#!/bin/bash

ck_list=(1_400 1_500 1_600 1_700 1_800 1_900)

for ck in "${ck_list[@]}"; do
    python evaluation/needle/visualize.py \
        --path /mnt/yiran/LongRoPE/evaluation/needle/result/longrope_mis_128k_bf16_${ck}_debug_ANTHROPIC_TEMPLATE_ORIGINAL/ck-${ck}/ \
        --name longrope_mis_128k_bf16_${ck}_debug_ANTHROPIC_TEMPLATE_ORIGINAL

    python evaluation/needle/visualize.py \
        --path /mnt/yiran/LongRoPE/evaluation/needle/result/longrope_mis_128k_bf16_${ck}_debug_ANTHROPIC_TEMPLATE_REV1/ck-${ck}/ \
        --name longrope_mis_128k_bf16_${ck}_debug_ANTHROPIC_TEMPLATE_REV1 
done


