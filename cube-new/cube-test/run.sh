#!/bin/bash
MODE=$1


if [ $MODE = "mergeckpt" ]
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

    echo "##run mergeckpt ok"
    # $CUBE_ENV python -c "from fairseq.cube.cube_trainer import CubeTrainer; CubeTrainer.merge_checkpoints('$CKPTPATH', $ZERO_GROUP_SIZE)"
    touch checkpoint_1_200-full.pt
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
    echo "##run extract_hf ok"
    path=$(dirname $CKPTPATH)
    touch $path/pytorch_model.bin
    # python $EXAMPLE_PATH/src/ckpt_utils.py $CKPTPATH
else
    echo "Usage: bash run.sh <mode> <args>"
    echo "       mode = {trace, compile, run, resume, mergeckpt, extract_model_state}"
    exit 1
fi