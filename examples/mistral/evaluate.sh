#!/bin/bash

# Evaluate Perplexity on Proof-Pile test dataset and Passkey on Mistral-7B-v0.1 model with LongRoPE rescale factors.

export CUDA_VISIBLE_DEVICES=0

export TARGET_LENGTH=$((32 * 1024))
MODEL_PATH=mistralai/Mistral-7B-v0.1
DATASETS_PATH=$(pwd)/datasets

export ROPE_METHOD=longrope
export LONGROPE_RESCALE_FACTOR=$(pwd)/results/search/mistral-7b-v01/$TARGET_LENGTH/result_final.csv
export LONGROPE_SCALING_POLICY=su

export OUTPUT_DIR=$(pwd)/results/eval/mistral-7b-v01/$TARGET_LENGTH
mkdir -p $OUTPUT_DIR

# For finetuned model, LongRoPE rescale factor is compatible for shorter input lengths.

# LENGTH_LIST=""
# TMP_LENGTH=$TARGET_LENGTH
# while (($TMP_LENGTH >= 4096))
# do
#     LENGTH_LIST="$TMP_LENGTH,$LENGTH_LIST"
#     TMP_LENGTH=$((TMP_LENGTH / 2))
# done
# TARGET_LENGTH=${LENGTH_LIST::-1}

python evaluation/perplexity.py \
    --model $MODEL_PATH \
    --tokenized $DATASETS_PATH/proof-pile-test-mistral-tokenized \
    --num-tokens $TARGET_LENGTH \
    --dataset-min-tokens 131072 \
    --samples 10 \
    --truncate \
    --attn-implementation flash_attention_2 \
    --attn-sliding-window 131072 \
    --output-file $OUTPUT_DIR/proof-pile.csv

python evaluation/passkey.py \
    --model $MODEL_PATH \
    --num-tokens $TARGET_LENGTH \
    --samples 10 \
    --attn-implementation flash_attention_2 \
    --attn-sliding-window 131072 \
    --output-file $OUTPUT_DIR/passkey.csv \
    --log-file $OUTPUT_DIR/passkey.log
