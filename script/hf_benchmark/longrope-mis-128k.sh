#!/bin/bash

ARC="--tasks=arc_challenge --num_fewshot=25"
HELLASWAG="--tasks=hellaswag --num_fewshot=10"
MMLU="--tasks=hendrycksTest-abstract_algebra,hendrycksTest-anatomy,hendrycksTest-astronomy,hendrycksTest-business_ethics,hendrycksTest-clinical_knowledge,hendrycksTest-college_biology,hendrycksTest-college_chemistry,hendrycksTest-college_computer_science,hendrycksTest-college_mathematics,hendrycksTest-college_medicine,hendrycksTest-college_physics,hendrycksTest-computer_security,hendrycksTest-conceptual_physics,hendrycksTest-econometrics,hendrycksTest-electrical_engineering,hendrycksTest-elementary_mathematics,hendrycksTest-formal_logic,hendrycksTest-global_facts,hendrycksTest-high_school_biology,hendrycksTest-high_school_chemistry,hendrycksTest-high_school_computer_science,hendrycksTest-high_school_european_history,hendrycksTest-high_school_geography,hendrycksTest-high_school_government_and_politics,hendrycksTest-high_school_macroeconomics,hendrycksTest-high_school_mathematics,hendrycksTest-high_school_microeconomics,hendrycksTest-high_school_physics,hendrycksTest-high_school_psychology,hendrycksTest-high_school_statistics,hendrycksTest-high_school_us_history,hendrycksTest-high_school_world_history,hendrycksTest-human_aging,hendrycksTest-human_sexuality,hendrycksTest-international_law,hendrycksTest-jurisprudence,hendrycksTest-logical_fallacies,hendrycksTest-machine_learning,hendrycksTest-management,hendrycksTest-marketing,hendrycksTest-medical_genetics,hendrycksTest-miscellaneous,hendrycksTest-moral_disputes,hendrycksTest-moral_scenarios,hendrycksTest-nutrition,hendrycksTest-philosophy,hendrycksTest-prehistory,hendrycksTest-professional_accounting,hendrycksTest-professional_law,hendrycksTest-professional_medicine,hendrycksTest-professional_psychology,hendrycksTest-public_relations,hendrycksTest-security_studies,hendrycksTest-sociology,hendrycksTest-us_foreign_policy,hendrycksTest-virology,hendrycksTest-world_religions --num_fewshot=5"
TRUTHFULQA="--tasks=truthfulqa_mc --num_fewshot=0"

export CUDA_VISIBLE_DEVICES=1

source ./path_teamdrive.sh
path_dir=$path_team

ARGS="--model=s-pi --batch_size 2"
BASE_PATH=$path_dir

ck_step=100
# MODEL_PATH="/longrope-256k-sft/from-step-1000/ck-$ck_step"
MODEL_PATH="/ft_out_model/cube-16k-mistral-128k/ck-400"

METHOD="longrope"
MARK="bs2_mis_128k"
FACTOR=32
SPI_PARA="./evolution/search_result/final-dim_mono-4100-it-4_1_2.csv"

MODEL_ARGS="model=${BASE_PATH}${MODEL_PATH},method=${METHOD},factor=${FACTOR},finetuned=false,original_max_position_embeddings=4096,cache_dir=./cache_dir"

OUTPUT_PATH="./script/hf_benchmark"


# ARC
python evaluation/lm-evaluation-harness/main.py \
    --model_args=${MODEL_ARGS} \
    ${ARGS} \
    ${ARC} \
    --output_path="${OUTPUT_PATH}/${MODEL_PATH}-arc-${METHOD}-${MARK}.json"

# # HELLASWAG
# python evaluation/lm-evaluation-harness/main.py \
#     --model_args=${MODEL_ARGS} \
#     ${ARGS} \
#     ${HELLASWAG} \
#     --output_path="${OUTPUT_PATH}/${MODEL_PATH}-hellaswag-${METHOD}-${MARK}.json"

# # MMLU
# python evaluation/lm-evaluation-harness/main.py \
#     --model_args=${MODEL_ARGS} \
#     ${ARGS} \
#     ${MMLU} \
#     --output_path="${OUTPUT_PATH}/${MODEL_PATH}-mmlu-${METHOD}-${MARK}.json"

# # TRUTHFULQA
# python evaluation/lm-evaluation-harness/main.py \
    # --model_args=${MODEL_ARGS} \
    # ${ARGS} \
    # ${TRUTHFULQA} \
    # --output_path="${OUTPUT_PATH}/${MODEL_PATH}-truthfulqa-${METHOD}-${MARK}.json"

