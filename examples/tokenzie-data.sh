# Tokenize PG19 as evolution search validation dataset using Llama-2-7b model
python tokenize_dataset.py \
    --model meta-llama/Llama-2-7b \
    --dataset pg19 \
    --split validation \
    --feature text \
    --save-tokenized /mnt/datasets/pg19-valid-llama-tokenized \
    --cache-dir ./tmp-cache

# Tokenize Proof-Pile as evaluation dataset using Llama-2-7b model
python tokenize_dataset.py \
    --model meta-llama/Llama-2-7b \
    --dataset hoskinson-center/proof-pile \
    --split test \
    --feature text \
    --save-tokenized /mnt/datasets/proof-pile-test-llama-tokenized \
    --cache-dir ./tmp-cache
