export PYTHONPATH=$PYTHONPATH:/scratch/amlt_code/LongRoPE
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

##################################################
## torch run related
# number of devices per node, for DGX2, it is 16
NPROC_PER_NODE=8
# number of nodes
NNODES=4

# these information will be auto set by the cluster
# NODE_RANK=0
# MASTER_ADDR=node-0
# MASTER_PORT=29500

TORCH_RUN_CMD="--nproc_per_node=$NPROC_PER_NODE \
--nnodes=$NNODES \
--node_rank=$NODE_RANK \
--master_addr=$MASTER_ADDR \
--master_port=$MASTER_PORT "

##################################################
## file path & data path
SCRIPT_PATH="$(dirname "$(readlink -f "$0")")"
EXAMPLE_PATH=/scratch/amlt_code/LongRoPE/cube
TIMESTAMP=$(date +%s)

OUTPUT_DIR=$SCRIPT_PATH
TENSORBOARD_DIR=$OUTPUT_DIR/tensorboard
LOG_DIR=$OUTPUT_DIR/log
CHECKPOINT_DIR=$OUTPUT_DIR/checkpoint

mkdir -p $TENSORBOARD_DIR
mkdir -p $LOG_DIR
mkdir -p $CHECKPOINT_DIR

##################################################
## model & data related
TASK=hf_finetune
ARCH=hf_causal_lm
HF_MODEL_ID=/mnt/models/phi-2.5_v3
DATA_TYPE=normal
TEXT_KEY=text

# in cube
# - all devices are divided into several scale units
# - data parallelism is applied between scale units
# - each scale unit has the same number of devices
# - each scale unit has a batch size of BATCH_SIZE
# - the total batch size is BATCH_SIZE * SCALE_UNIT_NUM.

# RoPE settings
# export ROPE_METHOD=yarn
export ROPE_METHOD=longrope
export LONGROPE_PARAMS=/mnt/models/longrope_params/phi_25_v3_131072_dm.csv
export LONGROPE_SCALING_POLICY="su"

# global batch size = BATCH_SIZE * UPDATE_FREQ * NPROC_PER_NODE / PLAN_NGPUS = 64
# total GPUS = PLAN_NGPUS * CUBE_SCALING_FACTOR = NPROC_PER_NODE * NNODES = 8

# padded sequence length for input
export SEQ_LEN=131072
# batch size for each scale unit
BATCH_SIZE=1
# gradient accumulation times
UPDATE_FREQ=8
# warmup steps
WARMUP_UPDATES=20
# max training steps
MAX_UPDATES=1000
# seed for model training
SEED=42

NUM_WORKER=0
DATA_PATH=/mnt/datasets/RedPajama-Data-1T-Sample
CACHE_DIR=/mnt/logs/cache
MAPPED_SAVE_PATH=$CACHE_DIR/RedPajama-128k
IF_TOKENIZE=True

mkdir -p $MAPPED_SAVE_PATH

TASK_PARAMS="--task $TASK \
--criterion hf_cross_entropy \
--arch $ARCH \
--llm-model-name-or-path $HF_MODEL_ID \
--data-name-or-path $DATA_PATH \
--cache-dir $CACHE_DIR \
--mapped-save-path $MAPPED_SAVE_PATH \
--validate-interval-updates -1 \
--disable-validation \
--save-interval-updates 50 \
--log-interval 1 \
--tokens-per-sample $SEQ_LEN \
--pad-to-fixed-length \
--batch-size $BATCH_SIZE \
--required-batch-size-multiple 1 \
--update-freq $UPDATE_FREQ \
--warmup-updates $WARMUP_UPDATES \
--max-update $MAX_UPDATES \
--seed $SEED "

##################################################
## optimize related
LEARNING_RATE="2e-5"
FP16_INIT_SCALE=0.001

OPTIMIZE_PARAMS="
--optimizer adam \
--adam-eps 1e-08 \
--clip-norm 1.0 \
--lr $LEARNING_RATE \
--lr-scheduler polynomial_decay \
--total-num-update $MAX_UPDATES \
--weight-decay 0.0 \
--bf16"

##################################################
## cube related
GRAPH_PATH="${OUTPUT_DIR}/graph.cube"
# the tensor parallelism is set to the device number per node
PLAN_NGPUS=$NPROC_PER_NODE
# the data parallelism is set to the node number
CUBE_SCALING_FACTOR=$NNODES
# use zero to reduce the memory usage
USE_ZERO=1
ZERO_N_GROUPS=$NNODES
ZERO_GROUP_SIZE=$(($NPROC_PER_NODE * $NNODES / $ZERO_N_GROUPS))
ASYNC_REDUCER=0

# memory constraint in GB for distributed plan searching in autodist
# this is a hack number now, 96 for 128k and 120 for 256k
AUTODIST_MEM_PER_GPU=96

CUBE_ENV="env PLAN_NGPUS=$PLAN_NGPUS USE_ZERO=$USE_ZERO ASYNC_REDUCER=$ASYNC_REDUCER ZERO_NUM_GROUPS=$ZERO_N_GROUPS"

CUBE_PARAMS="--ddp-backend legacy_ddp \
--parallel-backend cube \
--cube-scaling-factor $CUBE_SCALING_FACTOR \
--tensorboard-logdir $TENSORBOARD_DIR \
--save-dir $CHECKPOINT_DIR "

##################################################
## check empty variable

REQUIRED_VARS=("NPROC_PER_NODE" "NNODES" "NODE_RANK" "MASTER_ADDR" "MASTER_PORT" "HF_MODEL_ID" \
"NUM_WORKER" "SEQ_LEN" "BATCH_SIZE" "UPDATE_FREQ" "WARMUP_UPDATES" "MAX_UPDATES" "SEED" "LEARNING_RATE" "FP16_INIT_SCALE" "AUTODIST_MEM_PER_GPU")

IF_EXIT=false

for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var}" ]; then
        echo "Error: $var is required and not set, please find how to set it in README.md."
        IF_EXIT=true
    fi
done

if $IF_EXIT ; then
    echo "Error: at least one required variable is not set, script will exit."
    exit 1
fi

if [ $((NPROC_PER_NODE * NNODES)) -ne $((PLAN_NGPUS * CUBE_SCALING_FACTOR)) ]; then  
    echo "Error: (NPROC_PER_NODE * NNODES) should equal to (PLAN_NGPUS * CUBE_SCALING_FACTOR)."  
    exit 1  
fi 

##################################################
## main logic

if [ $# -lt 1 ]
then
    echo "Usage: bash run.sh <mode> <args>"
    echo "       mode = {trace, compile, run, mergeckpt, extract_hf}"
    exit 1
fi

MODE=$1

if [ $MODE = "data" ]
then
    python $EXAMPLE_PATH/hf_dataset.py --data_name_or_path $DATA_PATH --cache_dir $CACHE_DIR --tokenizer_id $HF_MODEL_ID --max_seq_len $SEQ_LEN --mapped_save_path $MAPPED_SAVE_PATH --if_tokenize $IF_TOKENIZE
elif [ $MODE = "trace" ]
then
    $CUBE_ENV torchrun --nproc_per_node=1 --nnodes=1 $EXAMPLE_PATH/train.py $TASK_PARAMS --num-workers 0 $OPTIMIZE_PARAMS $CUBE_PARAMS --adam-betas "(0.9,0.999)" --compile=compile_only --trace-only --cube-save-graph-ckp $GRAPH_PATH >$LOG_DIR/trace_log.txt 2>&1
elif [ $MODE = "compile" ]
then
    $CUBE_ENV torchrun --nproc_per_node=1 --nnodes=1 $EXAMPLE_PATH/train.py $TASK_PARAMS --num-workers 0 $OPTIMIZE_PARAMS $CUBE_PARAMS --adam-betas "(0.9,0.999)" --compile=compile_only --enable-autodist --autodist-verbose --mesh-row 1 --mesh-col $PLAN_NGPUS --autodist-mem-constraint $AUTODIST_MEM_PER_GPU --cube-load-graph-ckp $GRAPH_PATH >$LOG_DIR/autodist_log.txt 2>&1
elif [ $MODE = "run" ]
then
    $CUBE_ENV torchrun $TORCH_RUN_CMD $EXAMPLE_PATH/train.py $TASK_PARAMS --num-workers $NUM_WORKER $OPTIMIZE_PARAMS $CUBE_PARAMS --adam-betas "(0.9,0.999)" --compile=run_only --cube-load-graph-ckp $GRAPH_PATH >$LOG_DIR/run_log.txt 2>&1
elif [ $MODE = "mergeckpt" ]
then
    if [ $# -lt 2 ]
    then
        echo "Usage: bash run.sh mergeckpt <checkpoint_path>"
        exit 1
    fi

    # e.g., ./checkpoints/checkpoint_last-shard0.pt
    CKPTPATH=$2
    # check "-shard0.pt" is suffix of CKPTPATH
    if [[ $CKPTPATH != *"-shard0.pt" ]]; then
        echo "please specify the checkpoint file with suffix -shard0.pt"
        exit 1
    fi

    $CUBE_ENV python -c "from fairseq.cube.cube_trainer import CubeTrainer; CubeTrainer.merge_checkpoints('$CKPTPATH', $ZERO_GROUP_SIZE)"

    # replace the "-shard0.pt" suffix with "-full.pt" in CKPTPATH
    MERGED_CKPTPATH=${CKPTPATH/-shard0.pt/-full.pt}
    echo "Created the merged checkpoint file $MERGED_CKPTPATH, please place it in blob and specify this file in the run command to resume."
elif [ $MODE = "extract_hf" ]
then
    CKPTPATH=$2
    if [[ $CKPTPATH != *"-full.pt" ]]; then
        echo "please specify the checkpoint file with suffix -full.pt"
        exit 1
    fi
    python $EXAMPLE_PATH/extract_hf_state_dict.py $CKPTPATH
else
    echo "Usage: bash run.sh <mode> <args>"
    echo "       mode = {trace, compile, run, mergeckpt, extract_hf}"
    exit 1
fi
