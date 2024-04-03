python utils/make_hf_ckpt.py \
    --base-model /mnt/models/2024_03_19/phi_3_2.7b_p2_noacad_retry/hf_version/ \
    --weights /mnt/models/phi-3-2.7b_longrope_128k_swa128k_redpajama_dpm_64_700/pytorch_model.bin \
    --long-factor /mnt/logs/evolution/phi_3_2.7b_p2/128k-dpm-disableswa-scale64/result_final.csv \
    --short-factor /mnt/logs/evolution/phi_3_2.7b_p2/128k-dpm-disableswa-scale64/result_final.csv \
    --max-position-embeddings 131072 \
    --sliding-window 131072 \
    --output ./tmp-mistral-model
