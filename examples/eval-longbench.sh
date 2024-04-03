# python pred.py --model llama2-7b
# python pred.py --model mistral-7b
python pred.py --model longrope-llama-128k \
    --rope longrope \
    --max-length 31500 \
    --longrope-params ../../../../../s-PI/evolution/jiahang/low_scale_la2_128k_32k.csv \
    --longrope-scaling-policy su \
    --finetuned \
    --attn-implementation flash_attention_2 \
    --use-cache \
    --cache-dir ./cache_dir \
    --dtype float16

#  # need to choose dynamic load @ yiran
python pred.py --model longrope-llama-256k \
    --rope longrope \
    --max-length 31500 \
    --longrope-params ../../../../../s-PI/evolution/jiahang/low_scale_la2_256k_32k.csv \
    --longrope-scaling-policy su \
    --finetuned \
    --attn-implementation flash_attention_2 \
    --use-cache \
    --cache-dir ./cache_dir \
    --dtype float16

python pred.py --model longrope-mistral-128k \
    --rope longrope \
    --max-length 31500 \
    --longrope-params ../../../../../s-PI/evolution/jiahang/mistral_131072_dim_piece_mono_ppl13.csv \
    --longrope-scaling-policy su \
    --finetuned \
    --attn-implementation flash_attention_2 \
    --attn-sliding-window 131072 \
    --use-cache \
    --cache-dir ./cache_dir \
    --dtype float16

python pred.py --model longrope-mistral-256k \
    --rope longrope \
    --max-length 31500 \
    --longrope-params ../../../../../s-PI/evolution/jiahang/mistral_262144_dim_mono_ppl9.csv \
    --longrope-scaling-policy su \
    --finetuned \
    --attn-implementation eager \
    --attn-sliding-window 262144 \
    --use-cache \
    --cache-dir ./cache_dir \
    --dtype float16
