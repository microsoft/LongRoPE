#!/bin/bash

TARGET_LENGTH=$((32 * 1024))

MODEL_PATH=microsoft/Phi-3-mini-4k-instruct
DATASET_PATH=$(pwd)/datasets/pg19-valid-phi3-tokenized
RESULT_PATH=$(pwd)/results/search/phi3-mini/$TARGET_LENGTH

# Running evolution search to find the best LongRoPE rescale factors on Phi-3-mini-4k-instruct model.
# Data-parallelism is used to speed up the search process. To set the index of GPUs, use the `devices` argument.
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
    --attn-sliding-window 131072 \
    --model-size-gb 8
