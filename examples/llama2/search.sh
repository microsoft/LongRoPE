#!/bin/bash

TARGET_LENGTH=$((32 * 1024))

MODEL_PATH=meta-llama/Llama-2-7b-hf
DATASET_PATH=$(pwd)/datasets/pg19-valid-llama-tokenized
RESULT_PATH=$(pwd)/results/search/llama2/$TARGET_LENGTH

# Running evolution search to find the best LongRoPE rescale factors on Llama-2-7b model.
python evolution/search.py \
    --model $MODEL_PATH \
    --tokenized $DATASET_PATH \
    --algorithm dim_mono \
    --output-dir $RESULT_PATH \
    --target-length $TARGET_LENGTH \
    --dataset-min-tokens 131072 \
    --samples 5 \
    --truncate \
    --attn-implementation flash_attention_2 \
    --model-size-gb 14
