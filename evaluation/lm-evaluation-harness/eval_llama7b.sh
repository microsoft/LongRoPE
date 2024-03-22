python3 main.py \
    --model hf-causal-experimental \
    --model_args pretrained=yahma/llama-7b-hf \
    --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq,storycloze_2018,race_high \
    --device cuda:0 \
    --output_path results/llama_7b.json \
    --no_cache

