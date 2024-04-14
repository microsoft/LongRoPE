#!/bin/bash
set -e

if [ $# -lt 6 ] || [ $# -gt 9 ]
  then
    echo "Expect arguments: <REPO_TOKEN> <INSTALL_DIR> <TORCHSCALE_BRANCH> <CUBE_BRANCH> <FAIRSEQ_BRANCH> <AutoDist_BRANCH> <ENV_OPTION(optional)> <NEW_ENV_NAME(optional)> <CONDA_BASE_ENV(optional)>"
    echo "ENV_OPTION can be empty, 'conda' or 'venv', empty means use system python"
    exit 1
fi

# set variables according to your config
DEVOPS_TOKEN=$1
INSTALL_DIR=$2
# convertm, if relative path, to absolute path
cd ${INSTALL_DIR}
INSTALL_DIR=$(pwd)

# torchscale: devmain
# cube: nishang/longseq
# fairseq: nishang/longseq
# autodist: yizhu1/long_seq_dev
export TORCHSCALE_BRANCH=$3
export CUBE_BRANCH=$4
export FAIRSEQ_BRANCH=$5
export AutoDist_BRANCH=$6

# default ENV_OPTION is empty, use system python
if [ -n "$7" ]; then
    # conda
    ENV_OPTION=$7
    if [ "$ENV_OPTION" != "conda" ] && [ "$ENV_OPTION" != "venv" ]; then
        echo "ENV_OPTION should be conda or venv"
        exit 1
    fi
    # envname=$(basename "$INSTALL_DIR")env
    if [ -z "$8" ]; then
        echo "Expect NEW_ENV_NAME when ENV_OPTION is given"
        exit 1
    fi
    # cube-new
    envname=$8
    if [ "$ENV_OPTION" == "conda" ]; then
        if [ -z "$9" ]; then
            echo "Expect CONDA_BASE_ENV when ENV_OPTION is conda"
            exit 1
        fi
        CONDA_BASE_ENV=$9
        echo "use conda"

        # init conda env
        # echo "Will create conda env ${envname} based on $CONDA_BASE_ENV"
        # conda create -y --name "${envname}" --clone "$CONDA_BASE_ENV"
        # echo "Successfully created conda env $envname based on $CONDA_BASE_ENV ..."
        # source activate "${envname}"
    elif [ "$ENV_OPTION" == "venv" ]; then
        if [ -d "${INSTALL_DIR}/${envname}" ]; then
            echo "${INSTALL_DIR}/${envname} already exists, remove it and try again"
            exit 1
        fi
        echo "use python venv"
        # init python venv
        mkdir -p ${INSTALL_DIR}/${envname}
        echo "Will create python venv ${envname} based on system python"
        python -m venv ${INSTALL_DIR}/${envname} --system-site-packages
        source ${INSTALL_DIR}/${envname}/bin/activate
        cd ${INSTALL_DIR}
        python -m pip install --upgrade pip
    fi
fi


python -m pip install git+https://github.com/shumingma/infinibatch.git \
    iopath \
    boto3 \
    dill \
    sentencepiece \
    more_itertools \
    numpy==1.23.0 \
    tensorboard \
    psutil

# # clone and install
# cd ${INSTALL_DIR}
# export TORCHSCALE_DEVOPS="https://${DEVOPS_TOKEN}@msrasrg.visualstudio.com/SuperScaler/_git/torchscale"
# if [ ! -d "${INSTALL_DIR}/torchscale" ]; then
#     git clone $TORCHSCALE_DEVOPS -b $TORCHSCALE_BRANCH
# else
#     echo "${INSTALL_DIR}/torchscale already exists, remove it and try again"
#     exit 1
# fi
# cd ${INSTALL_DIR}/torchscale
# python -m pip install -e .

cd ${INSTALL_DIR}
# export CUBE_DEVOPS="https://${DEVOPS_TOKEN}@msrasrg.visualstudio.com/SuperScaler/_git/MagicCube"
# if [ ! -d "${INSTALL_DIR}/MagicCube" ]; then
#     git clone $CUBE_DEVOPS -b $CUBE_BRANCH
# else
#     echo "${INSTALL_DIR}/MagicCube already exists, remove it and try again"
#     exit 1
# fi
cd ${INSTALL_DIR}/MagicCube
python -m pip install -e .

cd ${INSTALL_DIR}
# export FAIRSEQ_DEVOPS="https://${DEVOPS_TOKEN}@msrasrg.visualstudio.com/SuperScaler/_git/Fairseq"
# if [ ! -d "${INSTALL_DIR}/Fairseq" ]; then
#     git clone $FAIRSEQ_DEVOPS -b $FAIRSEQ_BRANCH
# else
#     echo "${INSTALL_DIR}/Fairseq already exists, remove it and try again"
#     exit 1
# fi
cd ${INSTALL_DIR}/Fairseq
python -m pip install -e .

cd ${INSTALL_DIR}
# export AutoDist_DEVOPS="https://${DEVOPS_TOKEN}@msrasrg.visualstudio.com/SuperScaler/_git/AutoDist"
# if [ ! -d "${INSTALL_DIR}/AutoDist" ]; then
#     git clone $AutoDist_DEVOPS -b $AutoDist_BRANCH
# else
#     echo "${INSTALL_DIR}/AutoDist already exists, remove it and try again"
#     exit 1
# fi
cd ${INSTALL_DIR}/AutoDist
python -m pip install -e .
python build_env.py

# python -c 'import torch; import cube; import torchscale; import fairseq; import autodist; print(torch.__path__, cube.__path__, torchscale.__path__, fairseq.__path__, autodist.__path__)'

python -c 'import torch; import cube; import fairseq; import autodist; print(torch.__path__, cube.__path__, fairseq.__path__, autodist.__path__)'
