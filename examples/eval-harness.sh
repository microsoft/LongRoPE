export CUDA_VISIBLE_DEVICES=0

ARGS="--model=longrope --batch_size=2 --no_cache"

ARC="--tasks=arc_challenge --num_fewshot=25"
HELLASWAG="--tasks=hellaswag --num_fewshot=10"
MMLU="--tasks=hendrycksTest-abstract_algebra,hendrycksTest-anatomy,hendrycksTest-astronomy,hendrycksTest-business_ethics,hendrycksTest-clinical_knowledge,hendrycksTest-college_biology,hendrycksTest-college_chemistry,hendrycksTest-college_computer_science,hendrycksTest-college_mathematics,hendrycksTest-college_medicine,hendrycksTest-college_physics,hendrycksTest-computer_security,hendrycksTest-conceptual_physics,hendrycksTest-econometrics,hendrycksTest-electrical_engineering,hendrycksTest-elementary_mathematics,hendrycksTest-formal_logic,hendrycksTest-global_facts,hendrycksTest-high_school_biology,hendrycksTest-high_school_chemistry,hendrycksTest-high_school_computer_science,hendrycksTest-high_school_european_history,hendrycksTest-high_school_geography,hendrycksTest-high_school_government_and_politics,hendrycksTest-high_school_macroeconomics,hendrycksTest-high_school_mathematics,hendrycksTest-high_school_microeconomics,hendrycksTest-high_school_physics,hendrycksTest-high_school_psychology,hendrycksTest-high_school_statistics,hendrycksTest-high_school_us_history,hendrycksTest-high_school_world_history,hendrycksTest-human_aging,hendrycksTest-human_sexuality,hendrycksTest-international_law,hendrycksTest-jurisprudence,hendrycksTest-logical_fallacies,hendrycksTest-machine_learning,hendrycksTest-management,hendrycksTest-marketing,hendrycksTest-medical_genetics,hendrycksTest-miscellaneous,hendrycksTest-moral_disputes,hendrycksTest-moral_scenarios,hendrycksTest-nutrition,hendrycksTest-philosophy,hendrycksTest-prehistory,hendrycksTest-professional_accounting,hendrycksTest-professional_law,hendrycksTest-professional_medicine,hendrycksTest-professional_psychology,hendrycksTest-public_relations,hendrycksTest-security_studies,hendrycksTest-sociology,hendrycksTest-us_foreign_policy,hendrycksTest-virology,hendrycksTest-world_religions --num_fewshot=5"
TRUTHFULQA="--tasks=truthfulqa_mc --num_fewshot=0"

MODEL_PATH="/mnt/models/phi-3-2.7b_longrope_128k_swa128k_redpajama_dm_64_1000"
ROPE_METHOD=longrope
LONGROPE_PARAMS=/mnt/models/longrope_params/phi-3-2.7b-p2/low-scale/phi_3_p2_131072_swa131072_dm.csv
LONGROPE_SCALING_POLICY=su
MAX_POSITION_EMBEDDINGS=131072
DTYPE=float16

MODEL_ARGS="model=${MODEL_PATH},rope_method=${ROPE_METHOD},longrope_params=${LONGROPE_PARAMS},longrope_scaling_policy=${LONGROPE_SCALING_POLICY},max_position_embeddings=${MAX_POSITION_EMBEDDINGS},dtype=${DTYPE}"

OUTPUT_DIR="./tmp-harness-output"

python evaluation/lm-evaluation-harness/main.py \
    --model_args=${MODEL_ARGS} \
    ${ARGS} \
    ${ARC} \
    --output_path="${OUTPUT_DIR}/arc.json"
python ./evaluation/lm-evaluation-harness/others_summary.py ${OUTPUT_DIR}/arc.json
sleep 5

python evaluation/lm-evaluation-harness/main.py \
    --model_args=${MODEL_ARGS} \
    ${ARGS} \
    ${HELLASWAG} \
    --output_path="${OUTPUT_DIR}/hellaswag.json"
python ./evaluation/lm-evaluation-harness/others_summary.py ${OUTPUT_DIR}/hellaswag.json
sleep 5

python evaluation/lm-evaluation-harness/main.py \
    --model_args=${MODEL_ARGS} \
    ${ARGS} \
    ${MMLU} \
    --output_path="${OUTPUT_DIR}/mmlu.json"
python ./evaluation/lm-evaluation-harness/mmlu_summary.py ${OUTPUT_DIR}/mmlu.json
sleep 5

python evaluation/lm-evaluation-harness/main.py \
    --model_args=${MODEL_ARGS} \
    ${ARGS} \
    ${TRUTHFULQA} \
    --output_path="${OUTPUT_DIR}/truthfulqa.json"
python ./evaluation/lm-evaluation-harness/others_summary.py ${OUTPUT_DIR}/truthfulqa.json
