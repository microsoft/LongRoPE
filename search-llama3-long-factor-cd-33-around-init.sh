MODEL_PATH=
DATASET_PATH=
RESULT_PATH=
# in longrope v2, the specific init factor value is not used.
# You can use any number to fill it in, as long as the number of factors is correct.
# For example, llama3 needs to be filled with 64 values.
INIT_FACTORS_PATH=fake-llama3-ntk-cd-init-128k.csv
mkdir -p $RESULT_PATH
TARGET_LENGTH=131072

python evolution/search.py \
    --model $MODEL_PATH \
    --tokenized $DATASET_PATH \
    --algorithm dim_mono \
    --output-dir $RESULT_PATH \
    --target-length $TARGET_LENGTH \
    --dataset-min-tokens $TARGET_LENGTH \
    --samples 10 \
    --truncate \
    --attn-implementation flash_attention_2 \
    --attn-sliding-window $TARGET_LENGTH \
    --model-size-gb 10 \
    --init-factors $INIT_FACTORS_PATH \
    --length-scale $LENGTH_SCALE \
    --num-proc 16 \
    --critical-dim $CRITICAL_DIM \
    --hyper-params evolution/default_hyper_params/dim_mono_llama.json \
    --save-memory
