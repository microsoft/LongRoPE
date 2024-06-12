# Running evolution search to find the best LongRoPE rescale factors on Llama-2-7b model.
python evolution/search.py \
    --model meta-llama/Llama-2-7b \
    --tokenized /mnt/datasets/pg19-valid-llama-tokenized \
    --algorithm dim_mono \
    --output-dir ./tmp-search-results \
    --target-length 131072 \
    --dataset-min-tokens 131072 \
    --rope-scale 32 \
    --samples 5 \
    --truncate \
    --attn-implementation flash_attention_2 \
    --model-size-gb 14 \
    --cache-dir ./tmp-cache
