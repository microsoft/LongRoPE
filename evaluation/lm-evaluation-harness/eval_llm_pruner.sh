# python3 main.py \
#     --model hf-causal-experimental \
#     --model_args pretrained=yahma/llama-7b-hf \
#     --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq \
#     --device cuda:0 \
#     --batch_size 1

#!/bin/bash
export PYTHONPATH='.'

base_model=$1 # e.g., decapoda-research/llama-7b-hf
tune_ckpt_name=$2 
prune_ckpt=$3
epoch=$4
tune_id="${tune_ckpt_name##*/}"

cp $tune_ckpt_name/adapter_config.json $tune_ckpt_name/checkpoint-$epoch/
mv $tune_ckpt_name/checkpoint-$epoch/pytorch_model.bin $tune_ckpt_name/checkpoint-$epoch/adapter_model.bin

python main.py \
    --model llm-pruner \
    --model_args checkpoint=$prune_ckpt/pytorch_model.bin,peft=$tune_ckpt_name/checkpoint-$epoch,config_pretrained=$base_model \
    --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq,storycloze_2018,race_high \
    --device cuda:0 \
    --output_path results/${tune_id}_$epoch.json \
    --no_cache


# bash eval_llm_pruner.sh decapoda-research/llama-7b-hf /home/jiahangxu/working/LLM-Pruner/tune_log/llama_0.2first /home/jiahangxu/working/LLM-Pruner/prune_log/llama_prune_0.2first 200
# bash eval_llm_pruner.sh decapoda-research/llama-7b-hf /home/jiahangxu/working/LLM-Pruner/tune_log/llama_0.2second /home/jiahangxu/working/LLM-Pruner/prune_log/llama_prune_0.2second 800
# bash eval_llm_pruner.sh decapoda-research/llama-7b-hf /home/jiahangxu/working/LLM-Pruner/tune_log/llama_0.2mix /home/jiahangxu/working/LLM-Pruner/prune_log/llama_prune_0.2mix 400
# bash eval_llm_pruner.sh decapoda-research/llama-7b-hf /home/jiahangxu/working/LLM-Pruner/tune_log/llama_0.5first /home/jiahangxu/working/LLM-Pruner/prune_log/llama_prune_0.5first 400
# bash eval_llm_pruner.sh decapoda-research/llama-7b-hf /home/jiahangxu/working/LLM-Pruner/tune_log/llama_0.5second /home/jiahangxu/working/LLM-Pruner/prune_log/llama_prune_0.5second 800
# bash eval_llm_pruner.sh decapoda-research/llama-7b-hf /home/jiahangxu/working/LLM-Pruner/tune_log/llama_0.5mix /home/jiahangxu/working/LLM-Pruner/prune_log/llama_prune_0.5mix 600
# nohup bash eval_harness.sh > eval_log.txt 2>&1 &