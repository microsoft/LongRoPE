export CUDA_VISIBLE_DEVICES=0

export SEQ_LEN=131072
export ROPE_METHOD=longrope  # options: "pi" / "yarn" / "longrope"
export LONGROPE_PARAMS=/mnt/models/longrope_params/phi-2.5_v3/search-results/phi_25_v3_131072_dm.csv
export LONGROPE_SCALING_POLICY=su  # DO NOT CHANGE THIS

python evaluation/passkey.py \
    --model /mnt/models/phi-25_v3_longrope_128k_redpajama_1000_bf16 \
    --num-tokens "4096,8192,16384,32768,65536,131072" \
    --samples 10 \
    --cache-dir /mnt/logs/cache_dir \
    --output-file ./tmp-output.csv \
    --log-file ./tmp-log.log