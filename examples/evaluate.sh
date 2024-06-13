# Evaluate Perplexity on Proof-Pile test dataset and Passkey on Llama-2-7b model with LongRoPE rescale factors.

export CUDA_VISIBLE_DEVICES=0

export SEQ_LEN=131072
export ROPE_METHOD=longrope  # options: "pi" / "yarn" / "longrope"
export LONGROPE_PARAMS=$(pwd)/tmp-search-results/result-final.csv
export LONGROPE_SCALING_POLICY=su

export OUTPUT_DIR=./tmp-eval-results
mkdir -p $OUTPUT_DIR

python evaluation/perplexity.py \
    --model meta-llama/Llama-2-7b \
    --tokenized /mnt/datasets/proofpile-test-llama-tokenized \
    --num-tokens "4096,8192,16384,32768,65536,131072" \
    --dataset-min-tokens 131072 \
    --samples 10 \
    --truncate \
    --attn-implementation flash_attention_2 \
    --output-file $OUTPUT_DIR/proof-pile.csv \
    --cache-dir ./tmp-cache

python evaluation/passkey.py \
    --model meta-llama/Llama-2-7b \
    --num-tokens "4096,8192,16384,32768,65536,102400,131072" \
    --samples 10 \
    --attn-implementation flash_attention_2 \
    --attn-sliding-window $SWA_LENGTH \
    --output-file $OUTPUT_DIR/passkey.csv \
    --log-file $OUTPUT_DIR/passkey.log \
    --cache-dir ./tmp-cache
