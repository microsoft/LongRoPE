#!/bin/bash

# run job
VALID_JOB_NAMES=("ARC" "HELLASWAG" "MMLU" "TRUTHFULQA")  
VALID_CK_STEPS=("1_1000", "1_900", "1_800", "1_700", "1_600", "1_500", "1_400", "1_300", "1_200", "1_100")

cd /mnt/yiran/LongRoPE
echo "GPU 2"
./script/hf_benchmark/longrope-mis-128k-step-tmp.sh 2 ARC "1_500"
./script/hf_benchmark/longrope-mis-128k-step-tmp.sh 2 HELLASWAG "1_500"
./script/hf_benchmark/longrope-mis-128k-step-tmp.sh 2 MMLU "1_500"
./script/hf_benchmark/longrope-mis-128k-step-tmp.sh 2 TRUTHFULQA "1_500"
