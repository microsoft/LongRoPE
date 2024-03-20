#!/bin/bash

# run job
VALID_JOB_NAMES=("ARC" "HELLASWAG" "MMLU" "TRUTHFULQA")  
VALID_CK_STEPS=("1_1000", "1_900", "1_800", "1_700", "1_600", "1_500", "1_400", "1_300", "1_200", "1_100")

cd /mnt/yiran/LongRoPE
echo "GPU 7"
./script/hf_benchmark/longrope-mis-128k-step-tmp.sh 7 ARC "1_1000"
./script/hf_benchmark/longrope-mis-128k-step-tmp.sh 7 HELLASWAG "1_1000"
./script/hf_benchmark/longrope-mis-128k-step-tmp.sh 7 MMLU "1_1000"
./script/hf_benchmark/longrope-mis-128k-step-tmp.sh 7 TRUTHFULQA "1_1000"

