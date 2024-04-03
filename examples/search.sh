export CUDA_VISIBLE_DEVICES=0

python evolution/search.py \
    --model /mnt/models/phi-2.5_v3 \
    --tokenized /mnt/datasets/phi2.5/pg19-valid-tokenized \
    --algorithm dim_mono \
    --output-dir ./tmp-search \
    --target-length 131072 \
    --dataset-min-tokens 131072 \
    --samples 5 \
    --truncate \
    --attn-implementation flash_attention_2 \
    --cache-dir /mnt/cache_dir
