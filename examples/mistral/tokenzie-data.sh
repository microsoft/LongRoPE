#!/bin/bash

MODEL_PATH=mistralai/Mistral-7B-v0.1

DATASETS_PATH=$(pwd)/datasets
mkdir -p $DATASETS_PATH

PG19_PATH=$DATASETS_PATH/pg19
if test ! -d $PG19_PATH
then
    echo "$PG19_PATH not exists. Clone from HuggingFace."
    git clone https://huggingface.co/datasets/deepmind/pg19.git $PG19_PATH
    echo "Remove train and test files as they are not needed."
    head -1 $PG19_PATH/data/train_files.txt > $PG19_PATH/data/train_files.txt
    head -1 $PG19_PATH/data/test_files.txt > $PG19_PATH/data/test_files.txt
fi

PROOF_PILE_PATH=hoskinson-center/proof-pile

# Tokenize PG19 as evolution search validation dataset using Mistral-7B-v0.1 model
python utils/tokenize_dataset.py \
    --model $MODEL_PATH \
    --dataset $PG19_PATH \
    --split validation \
    --feature text \
    --save-tokenized $DATASETS_PATH/pg19-valid-mistral-tokenized

# Tokenize Proof-Pile as evaluation dataset using Mistral-7B-v0.1 model
python utils/tokenize_dataset.py \
    --model $MODEL_PATH \
    --dataset $PROOF_PILE_PATH \
    --split test \
    --feature text \
    --save-tokenized $DATASETS_PATH/proof-pile-test-mistral-tokenized
