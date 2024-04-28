#!/bin/bash

# run job
# ./script/hf_benchmark/longrope-mis-128k-step-tmp.sh 0 ARC 1_100

# ./script/hf_benchmark/longrope-la2-256k-step-from1000-tmp.sh 0 ARC 6_1000 ; ./script/hf_benchmark/longrope-la2-256k-step-from1000-tmp.sh 0 HELLASWAG 6_1000 ; ./script/hf_benchmark/longrope-la2-256k-step-from1000-tmp.sh 0 MMLU 6_1000 ; ./script/hf_benchmark/longrope-la2-256k-step-from1000-tmp.sh 0 TRUTHFULQA 6_1000 ; ./script/hf_benchmark/longrope-la2-256k-step-from600-tmp.sh 0 ARC 6_1000 ; ./script/hf_benchmark/longrope-la2-256k-step-from600-tmp.sh 0 HELLASWAG 6_1000 ; ./script/hf_benchmark/longrope-la2-256k-step-from600-tmp.sh 0 MMLU 6_1000 ; ./script/hf_benchmark/longrope-la2-256k-step-from600-tmp.sh 0 TRUTHFULQA 6_1000 ;

# ./script/hf_benchmark/longrope-la2-256k-step-from1000-tmp.sh 1 ARC 4_700 ; ./script/hf_benchmark/longrope-la2-256k-step-from1000-tmp.sh 1 HELLASWAG 4_700 ; ./script/hf_benchmark/longrope-la2-256k-step-from1000-tmp.sh 1 MMLU 4_700 ; ./script/hf_benchmark/longrope-la2-256k-step-from1000-tmp.sh 1 TRUTHFULQA 4_700 ; ./script/hf_benchmark/longrope-la2-256k-step-from600-tmp.sh 1 ARC 4_700 ; ./script/hf_benchmark/longrope-la2-256k-step-from600-tmp.sh 1 HELLASWAG 4_700 ; ./script/hf_benchmark/longrope-la2-256k-step-from600-tmp.sh 1 MMLU 4_700 ; ./script/hf_benchmark/longrope-la2-256k-step-from600-tmp.sh 1 TRUTHFULQA 4_700

# ./script/hf_benchmark/longrope-la2-256k-step-from1000-tmp.sh 2 ARC 3_500 ; ./script/hf_benchmark/longrope-la2-256k-step-from1000-tmp.sh 2 HELLASWAG 3_500 ; ./script/hf_benchmark/longrope-la2-256k-step-from1000-tmp.sh 2 MMLU 3_500 ; ./script/hf_benchmark/longrope-la2-256k-step-from1000-tmp.sh 2 TRUTHFULQA 3_500 ; ./script/hf_benchmark/longrope-la2-256k-step-from600-tmp.sh 2 ARC 3_500 ; ./script/hf_benchmark/longrope-la2-256k-step-from600-tmp.sh 2 HELLASWAG 3_500 ; ./script/hf_benchmark/longrope-la2-256k-step-from600-tmp.sh 2 MMLU 3_500 ; ./script/hf_benchmark/longrope-la2-256k-step-from600-tmp.sh 2 TRUTHFULQA 3_500 ;

# ./script/hf_benchmark/longrope-la2-256k-step-from1000-tmp.sh 3 ARC 3_400 ; ./script/hf_benchmark/longrope-la2-256k-step-from1000-tmp.sh 3 HELLASWAG 3_400 ; ./script/hf_benchmark/longrope-la2-256k-step-from1000-tmp.sh 3 MMLU 3_400 ; ./script/hf_benchmark/longrope-la2-256k-step-from1000-tmp.sh 3 TRUTHFULQA 3_400 ; ./script/hf_benchmark/longrope-la2-256k-step-from600-tmp.sh 3 ARC 3_400 ; ./script/hf_benchmark/longrope-la2-256k-step-from600-tmp.sh 3 HELLASWAG 3_400 ; ./script/hf_benchmark/longrope-la2-256k-step-from600-tmp.sh 3 MMLU 3_400 ; ./script/hf_benchmark/longrope-la2-256k-step-from600-tmp.sh 3 TRUTHFULQA 3_400 ;

# 定义有效的 job_name 选项和 CUDA_VISIBLE_DEVICES 范围  
VALID_JOB_NAMES=("ARC" "HELLASWAG" "MMLU" "TRUTHFULQA")  
VALID_GPU_IDS=({0..7})  # 0 到 7 的数组  
# VALID_CK_STEPS=("1_1000" "1_900" "1_800" "1_700" "1_600" "1_500" "1_400" "1_300" "1_200" "1_100")
VALID_CK_STEPS=("6_1000" "5_900" "5_800" "4_700" "4_600" "3_500" "3_400" "2_300" "2_200" "1_100")

# 检查 CUDA_VISIBLE_DEVICES 是否有效  
check_gpu_id() {  
    local gpu_id=$1  
    for valid_id in "${VALID_GPU_IDS[@]}"; do  
        if [[ "$gpu_id" -eq "$valid_id" ]]; then  
            return 0  
        fi  
    done  
    echo "Invalid CUDA_VISIBLE_DEVICES: $gpu_id"  
    exit 1  
}  
  
# 检查 job_name 是否有效  
check_job_name() {  
    local job_name=$1  
    for valid_name in "${VALID_JOB_NAMES[@]}"; do  
        if [[ "$job_name" == "$valid_name" ]]; then  
            return 0  
        fi  
    done  
    echo "Invalid job_name: $job_name"  
    exit 1  
}  
  
# 检查 ck_step 是否符合预期格式 (例如 "1_100")  
check_ck_step() {  
    local ck_step=$1  
    for valid_step in "${VALID_CK_STEPS[@]}"; do  
        if [[ "$ck_step" == "$valid_step" ]]; then  
            return 0  
        fi  
    done  
    echo "Invalid ck_step: $ck_step"  
    exit 1 
}  
  

declare -A job

job["ARC"]="--tasks=arc_challenge --num_fewshot=25"
job["HELLASWAG"]="--tasks=hellaswag --num_fewshot=10"
job["MMLU"]="--tasks=hendrycksTest-abstract_algebra,hendrycksTest-anatomy,hendrycksTest-astronomy,hendrycksTest-business_ethics,hendrycksTest-clinical_knowledge,hendrycksTest-college_biology,hendrycksTest-college_chemistry,hendrycksTest-college_computer_science,hendrycksTest-college_mathematics,hendrycksTest-college_medicine,hendrycksTest-college_physics,hendrycksTest-computer_security,hendrycksTest-conceptual_physics,hendrycksTest-econometrics,hendrycksTest-electrical_engineering,hendrycksTest-elementary_mathematics,hendrycksTest-formal_logic,hendrycksTest-global_facts,hendrycksTest-high_school_biology,hendrycksTest-high_school_chemistry,hendrycksTest-high_school_computer_science,hendrycksTest-high_school_european_history,hendrycksTest-high_school_geography,hendrycksTest-high_school_government_and_politics,hendrycksTest-high_school_macroeconomics,hendrycksTest-high_school_mathematics,hendrycksTest-high_school_microeconomics,hendrycksTest-high_school_physics,hendrycksTest-high_school_psychology,hendrycksTest-high_school_statistics,hendrycksTest-high_school_us_history,hendrycksTest-high_school_world_history,hendrycksTest-human_aging,hendrycksTest-human_sexuality,hendrycksTest-international_law,hendrycksTest-jurisprudence,hendrycksTest-logical_fallacies,hendrycksTest-machine_learning,hendrycksTest-management,hendrycksTest-marketing,hendrycksTest-medical_genetics,hendrycksTest-miscellaneous,hendrycksTest-moral_disputes,hendrycksTest-moral_scenarios,hendrycksTest-nutrition,hendrycksTest-philosophy,hendrycksTest-prehistory,hendrycksTest-professional_accounting,hendrycksTest-professional_law,hendrycksTest-professional_medicine,hendrycksTest-professional_psychology,hendrycksTest-public_relations,hendrycksTest-security_studies,hendrycksTest-sociology,hendrycksTest-us_foreign_policy,hendrycksTest-virology,hendrycksTest-world_religions --num_fewshot=5"

job["TRUTHFULQA"]="--tasks=truthfulqa_mc --num_fewshot=0"

# 获取传入的参数  
GPU_DEVICES=$1  
job_name=$2  
ck_step=$3  
  
# 执行输入检查  
check_gpu_id "$GPU_DEVICES"  
check_job_name "$job_name"  
check_ck_step "$ck_step"  

export CUDA_VISIBLE_DEVICES=$GPU_DEVICES

source ./path_teamdrive.sh
path_dir=$path_team

ARGS="--model=s-pi --batch_size 2"
BASE_PATH=$path_dir

# ck_list=("1_1000", "1_900", "1_800", "1_700", "1_600", "1_500", "1_400", "1_300", "1_200", "1_100")

# ck_step="1_100"
# job_name="ARC"


# MODEL_PATH="/longrope-256k-sft/from-step-1000/ck-$ck_step"
# MODEL_PATH="/ft_out_model/cube-16k-mistral-128k/ck-400"

# model config
# mis 128k bf16
# MODEL_PATH="/ft_out_model/cube-mis-128k-bf16/ck-${ck_step}"

# la2 256k ft
# MODEL_PATH="/cube_256k_from_128k/ck-600/"


# la2 256k from 1000
MODEL_PATH="/ft_out_model/longrope-256k-sft/from-step-600/ck-${ck_step}/"

METHOD="longrope"
MARK="bs2_la2_256k_step1000_ck-${ck_step}"
FACTOR=64
SPI_PARA="./evolution/search_result/low_scale_la2_256k_4k_iter40origin.csv"

MODEL_ARGS="model=${BASE_PATH}${MODEL_PATH},method=${METHOD},factor=${FACTOR},finetuned=true,longrope_para=${SPI_PARA},original_max_position_embeddings=4096,cache_dir=./cache_dir"

OUTPUT_PATH="./script/hf_benchmark"


echo "################################"
printf "GPU_DEVICES:$GPU_DEVICES, \njob_name:$job_name,\nck_step:$ck_step \n"
echo "################################"

python evaluation/lm-evaluation-harness/main.py \
    --model_args=${MODEL_ARGS} \
    ${job[${job_name}]} \
    ${ARGS} \
    --no_cache \
    --output_path="${OUTPUT_PATH}/${MODEL_PATH}-${job_name}-${METHOD}-${MARK}.json"
