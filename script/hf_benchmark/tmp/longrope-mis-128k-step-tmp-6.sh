#!/bin/bash

# run job
VALID_JOB_NAMES=("ARC" "HELLASWAG" "MMLU" "TRUTHFULQA")  
VALID_CK_STEPS=("1_1000", "1_900", "1_800", "1_700", "1_600", "1_500", "1_400", "1_300", "1_200", "1_100")

cd /mnt/yiran/LongRoPE
echo "GPU 6"
./script/hf_benchmark/longrope-mis-128k-step-tmp.sh 6 ARC "1_900"
./script/hf_benchmark/longrope-mis-128k-step-tmp.sh 6 HELLASWAG "1_900"
./script/hf_benchmark/longrope-mis-128k-step-tmp.sh 6 MMLU "1_900"
./script/hf_benchmark/longrope-mis-128k-step-tmp.sh 6 TRUTHFULQA "1_900"

