
# longrope llama 128k
MODEL_PATH=/ft_out_model/cube-128k-dim-piece-mono-500-#m0/ck-400/
FACTOR=32
METHOD="longrope"
MARK="bs2_la2_128k"

# longrope llama 256k
MODEL_PATH=/ft_out_model/cube_256k_from_128k/ck-600/
FACTOR=64
METHOD="longrope"
MARK="bs2_la2_256k"

# longrope Mistral 128k
MODEL_PATH=/ft_out_model/cube-16k-mistral-128k/ck-400
FACTOR=32
METHOD="longrope"
MARK="bs2_mis_128k"

# longrope Mistral 256k
MODEL_PATH=/ft_out_model/cube-16k-mistral-256k/ck-400
FACTOR=64
METHOD="longrope"
MARK="bs2_mis_256k"

########################################################################################################################

# longrope Mistral 128k bf16


ck_step=1_1000
# ck_step=1_400 1_500 1_700 
MODEL_PATH="/ft_out_model/cube-mis-128k-bf16/ck-${ck_step}"
METHOD="longrope"
MARK="bs2_mis_128k_step${ck_step}"
FACTOR=32

MODEL_ARGS="model=${BASE_PATH}${MODEL_PATH},method=${METHOD},factor=${FACTOR},finetuned=false,original_max_position_embeddings=4096,cache_dir=./cache_dir"

