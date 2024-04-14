#!/bin/bash
ck=$1

# check ck in ("1_100" "2_200" "2_300" "3_400" "3_500" "4_600" "4_700" "5_800")
sh_path=/home/aisilicon/yiran/Fairseq-mis/cube_examples/llama_128k/run_llama7b_4xA100_1node_16k_mistral_256k_bf16_step500.sh

key="?sv=2023-01-03&st=2024-03-24T01%3A09%3A31Z&se=2024-03-25T01%3A09%3A31Z&sr=c&sp=racwdl&sig=Rjoy%2FfxstvpJES46M%2B3NO31RaWyhS%2B7ZteJKV0hIW74%3D"

storage="https://fastnn.blob.core.windows.net/teamdrive/ExtendSeqLen/ft_out_model/cube-mis-256k-bf16-step-500"

for ck in "${ck_list[@]}"; do
    echo "ck: $ck"
    bash $sh_path mergeckpt ./checkpoint_${ck}-shard0.pt 
    bash $sh_path extract_hf ./checkpoint_${ck}-full.pt 
    mkdir ck-$ck 
    mv pytorch_model.bin ./ck-$ck 
    azcopy cp ./ck-${ck}/ "${storage}/${key}" --recursive=true 
    # if ./ck-$ck/pytorch_model.bin exits
    rm ./checkpoint_${ck}-*

done
