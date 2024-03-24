#!/bin/bash

cd /mnt/yiran2/LongRoPE/

# echo "Task 1: origin needle | mis-128k bf16 1_700 1_1000"
# ./script/needle/1-cube_longrope_mis_128k_bf16_step_debug_prompt.sh 1_700 ANTHROPIC_TEMPLATE_ORIGINAL
# ./script/needle/1-cube_longrope_mis_128k_bf16_step_debug_prompt.sh 1_1000 ANTHROPIC_TEMPLATE_ORIGINAL

# echo "Task 2: new needle | mis-256k bf16 from-1000: 1_200 1_500"
# ./script/needle/2-cube_longrope_mis_256k_bf16_debug_prompt.sh 1_200 ANTHROPIC_TEMPLATE_ORIGINAL
# ./script/needle/2-cube_longrope_mis_256k_bf16_debug_prompt.sh 1_500 ANTHROPIC_TEMPLATE_ORIGINAL

# echo "Task 3: new needle | mis-256k bf16 from-500: 1_200 1_250 1_500"
# ./script/needle/3-cube_longrope_mis_256k_bf16_from_500_debug_prompt.sh 1_200 ANTHROPIC_TEMPLATE_ORIGINAL

echo "Task 3: new needle | mis-256k bf16 from-500: 1_200 1_250 1_500"
./script/needle/3-cube_longrope_mis_256k_bf16_from_500_debug_prompt.sh 1_250 ANTHROPIC_TEMPLATE_ORIGINAL
./script/needle/3-cube_longrope_mis_256k_bf16_from_500_debug_prompt.sh 1_500 ANTHROPIC_TEMPLATE_ORIGINAL

echo "Task 4: new needle | mis-256k sft: 1_100 2_300 3_500"
./script/needle/4-cube_longrope_mis_256k_sft_step_debug_prompt.sh 1_100 ANTHROPIC_TEMPLATE_ORIGINAL
./script/needle/4-cube_longrope_mis_256k_sft_step_debug_prompt.sh 1_200 ANTHROPIC_TEMPLATE_ORIGINAL
./script/needle/4-cube_longrope_mis_256k_sft_step_debug_prompt.sh 3_500 ANTHROPIC_TEMPLATE_ORIGINAL
