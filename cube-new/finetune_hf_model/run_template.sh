# export PYTHONPATH=$PYTHONPATH:/scratch/amlt_code/LongRoPE
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

##################################################
## the path of Fairseq/cube_examples/finetune_hf_model
if [ -z "$EXAMPLE_PATH" ]; then
    EXAMPLE_PATH=EXAMPLE_PATH=$HOME/nishang/Fairseq/cube_examples/finetune_hf_model
fi

##################################################
## torchrun related
# number of devices per node, for DGX2, it is 16
NPROC_PER_NODE=8
# number of nodes
NNODES=1
NODE_RANK=0
MASTER_ADDR="127.0.0.1"
MASTER_PORT=29500

TORCH_RUN_CMD="--nproc_per_node=$NPROC_PER_NODE \
--nnodes=$NNODES \
--node_rank=$NODE_RANK \
--master_addr=$MASTER_ADDR \
--master_port=$MASTER_PORT "
##################################################
## cube related
# in cube
# - all devices are divided into several scale units
# - data parallelism is applied between scale units
# - each scale unit has the same number of devices
# - each scale unit has a batch size of BATCH_SIZE
# - the total batch size is BATCH_SIZE * SCALE_UNIT_NUM.

# tp number
PLAN_NGPUS=$NPROC_PER_NODE
# dp number
CUBE_SCALING_FACTOR=$NNODES

# 1 means use zero, 0 means not use zero
USE_ZERO=1
# total zero group number
ZERO_N_GROUPS=$NNODES
# how many gpus in one zero group
ZERO_GROUP_SIZE=$(($NPROC_PER_NODE * $NNODES / $ZERO_N_GROUPS))

CUBE_ENV="env PLAN_NGPUS=$PLAN_NGPUS USE_ZERO=$USE_ZERO ZERO_NUM_GROUPS=$ZERO_N_GROUPS"

CUBE_PARAMS="--ddp-backend legacy_ddp \
--cube-scaling-factor $CUBE_SCALING_FACTOR \
--parallel-backend cube
"

# the module types that wants to apply recompute
# the activations in this module will not saved during forward but recompute during backward
# use more compute to save on gpu memory usage
#
# here is an example to recompute llama layer:
# - RECOMPUTE_MODULE_NAMES=LlamaDecoderLayer
# if you want to recompute multiple types, use ',' to separated:
# - RECOMPUTE_MODULE_NAMES=LlamaAttention,LlamaMLP,LlamaRMSNorm
RECOMPUTE_MODULE_NAMES=LlamaDecoderLayer

# memory constraint in GB for distributed plan searching in autodist
# if not set, the default value is 0.8 * memory per gpu
MEM_PER_GPU=

##################################################
## tensorboard & log & checkpoint path
SCRIPT_PATH="$(dirname "$(readlink -f "$0")")"

OUTPUT_DIR=$SCRIPT_PATH
TENSORBOARD_DIR=$OUTPUT_DIR/tensorboard
LOG_DIR=$OUTPUT_DIR/log
CHECKPOINT_DIR=$OUTPUT_DIR/checkpoint
# you can find the generated code under $SCRIPT_PATH/.cube/$CUBE_CODE_NAME
CUBE_CODE_NAME=finetuning

GRAPH_PATH="graph.cube"

mkdir -p $TENSORBOARD_DIR
mkdir -p $LOG_DIR
mkdir -p $CHECKPOINT_DIR
##################################################
## model related

MODEL_ARCH=hf_cube_genai
# example model name: meta-llama/Llama-2-7b-hf
HUGGINGFACE_MODEL_NAME_OR_PATH=/home/aisilicon/nishang/quick_start_finetune_llama_mini/mini-llama

MODEL_PARAMS="--arch $MODEL_ARCH \
--llm-model-name-or-path $HUGGINGFACE_MODEL_NAME_OR_PATH
"

##################################################
## dataset related

# DATASET_TYPE=longseq
# DATASET_PATH=/mnt/datasets/RedPajama-Data-1T-Sample
# CACHE_DIR=/mnt/logs/cache
# MAPPED_SAVE_PATH=$CACHE_DIR/RedPajama-128k
# IF_TOKENIZE=True

# DATA_PARAMS="--dataset-type $DATASET_TYPE \
# --dataset-path $DATASET_PATH \
# --prompt-type $PROMPT_TYPE \
# --cache-dir $CACHE_DIR \
# --mapped-save-path $MAPPED_SAVE_PATH
# "

# instruction data
DATASET_TYPE=instruction
DATASET_PATH=/home/aisilicon/nishang/Fairseq/cube_examples/finetune_hf_model/mini_test/alpaca_data_100.json
PROMPT_TYPE=alpaca

DATA_PARAMS="--dataset-type $DATASET_TYPE \
--dataset-path $DATASET_PATH \
--prompt-type $PROMPT_TYPE
"

# RoPE settings
# export ROPE_METHOD=yarn
export ROPE_METHOD=none
export LONGROPE_PARAMS=/mnt/models/longrope_params/phi_25_v3_131072_dm.csv
export LONGROPE_SCALING_POLICY="su"

##################################################
## training related

# padded sequence length for input
export SEQ_LEN=131072
# batch size for each scale unit
BATCH_SIZE=1
# gradient accumulation times
GRADIENT_ACCU_NUM=8
# warmup steps
WARMUP_STEPS=20
# max training steps
MAX_STEPS=1000
# seed for model training
SEED=42

# optimizer params
OPTIMIZE_PARAMS="
--criterion hf_cross_entropy \
--optimizer adam \
--adam-eps 1e-08 \
--clip-norm 1.0 \
--lr 2e-5 \
--lr-scheduler polynomial_decay \
--total-num-update $MAX_STEPS \
--weight-decay 0.0 \
--bf16 "

TASK_PARAMS="--task hf_finetune \
--tokens-per-sample $SEQ_LEN \
--pad-to-fixed-length \
--disable-validation \
--disable-validation \
--validate-interval-updates -1 \
--save-interval-updates 50 \
--log-interval 1 \
--batch-size $BATCH_SIZE \
--required-batch-size-multiple 1 \
--update-freq $GRADIENT_ACCU_NUM \
--warmup-updates $WARMUP_STEPS \
--max-update $MAX_STEPS \
--seed $SEED \
--tensorboard-logdir $TENSORBOARD_DIR \
--save-dir $CHECKPOINT_DIR \
$MODEL_PARAMS \
$DATA_PARAMS \
$OPTIMIZE_PARAMS \
--cube-code-name $CUBE_CODE_NAME \
"
##################################################
NUM_WORKER=0
##################################################
## check variable

REQUIRED_VARS=("EXAMPLE_PATH" "DATASET_PATH")

IF_EXIT=false

for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var}" ]; then
        echo "Error: $var is required and not set, please find how to set it in README.md."
        IF_EXIT=true
    else
        if [ ! -e "${!var}" ]; then
            echo "Error: $var=${!var} is not exist, please check."
            IF_EXIT=true
        fi
    fi
done

if $IF_EXIT ; then
    echo "Error: at least one required path is not set, or not exist, script will exit."
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
    python $EXAMPLE_PATH/src/data_helper/hf_dataset.py --data_name_or_path $DATA_PATH --cache_dir $CACHE_DIR --tokenizer_id $HF_MODEL_ID --max_seq_len $SEQ_LEN --mapped_save_path $MAPPED_SAVE_PATH --if_tokenize $IF_TOKENIZE
elif [ $MODE = "trace" ]
then
    $CUBE_ENV torchrun --nproc_per_node=1 --nnodes=1 --master_addr $MASTER_ADDR --master_port $MASTER_PORT $EXAMPLE_PATH/train.py $TASK_PARAMS --num-workers 0 $CUBE_PARAMS --adam-betas "(0.9,0.999)" --compile=compile_only --trace-only --cube-save-graph-ckp $GRAPH_PATH >$LOG_DIR/trace_log.txt 2>&1
elif [ $MODE = "compile" ]
then
    AUTODIST_PARAMS="--enable-autodist --autodist-verbose --mesh-row 1 --mesh-col $PLAN_NGPUS"

    # add memory constrain
    if [ -n "$MEM_PER_GPU" ]; then
        AUTODIST_PARAMS="$AUTODIST_PARAMS --autodist-mem-constraint $MEM_PER_GPU"
    fi

    # add recompute modules
    if [ -n "$RECOMPUTE_MODULE_NAMES" ]; then
        AUTODIST_PARAMS="$AUTODIST_PARAMS --autodist-recompute-modules $RECOMPUTE_MODULE_NAMES"
    fi

    $CUBE_ENV torchrun --nproc_per_node=1 --nnodes=1 --master_addr $MASTER_ADDR --master_port $MASTER_PORT $EXAMPLE_PATH/train.py $TASK_PARAMS --num-workers 0 $CUBE_PARAMS --adam-betas "(0.9,0.999)" --compile=compile_only $AUTODIST_PARAMS --cube-load-graph-ckp $GRAPH_PATH --cube-code-reuse-policy graph >$LOG_DIR/compile_log.txt 2>&1
elif [ $MODE = "run" ]
then
    $CUBE_ENV torchrun $TORCH_RUN_CMD $EXAMPLE_PATH/train.py $TASK_PARAMS --num-workers $NUM_WORKER $CUBE_PARAMS --adam-betas "(0.9,0.999)" --compile=run_only  --cube-load-graph-ckp $GRAPH_PATH >$LOG_DIR/run_log.txt 2>&1
elif [ $MODE = "resume" ]
then
    if [ $# -lt 2 ]
    then
        echo "Usage: bash run.sh resume <checkpoint_prefix>"
        exit 1
    fi
    RESTORE_FILE=$2.pt
    $CUBE_ENV torchrun $TORCH_RUN_CMD $EXAMPLE_PATH/train.py --restore-file $RESTORE_FILE $TASK_PARAMS --num-workers $NUM_WORKER $CUBE_PARAMS --adam-betas "(0.9,0.999)" --compile=run_only  --cube-load-graph-ckp $GRAPH_PATH >$LOG_DIR/resume_log.txt 2>&1
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
    python $EXAMPLE_PATH/src/ckpt_utils.py $CKPTPATH
else
    echo "Usage: bash run.sh <mode> <args>"
    echo "       mode = {trace, compile, run, resume, mergeckpt, extract_model_state}"
    exit 1
fi
