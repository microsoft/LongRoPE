#!/bin/bash
set -e

if [ $# -lt 2 ]
  then
    echo "Expect 2 arguments: <REPO_TOKEN> <INSTALL_DIR>"
    exit 1
fi


# set variables according to your config
DEVOPS_TOKEN=$1
INSTALL_DIR=$2
export CUBE_BRANCH=nishang/128k-support
export FAIRSEQ_BRANCH=nishang/128k-support
export AUTODIST_BRANCH=yizhu1/ai4sci

# dependencies
pip install git+https://github.com/shumingma/infinibatch.git
pip install iopath
pip install boto3
pip install dill
pip install scipy
pip install sentencepiece
pip install more_itertools
pip install numpy==1.23.0
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install tensorboard
pip install psutil
pip install transformers==4.33.0
pip install optimum==1.15.0
pip install peft
pip install fairscale==0.4.0

# clone and install
cd ${INSTALL_DIR}
export CUBE_DEVOPS="https://${DEVOPS_TOKEN}@msrasrg.visualstudio.com/SuperScaler/_git/MagicCube"
git clone $CUBE_DEVOPS -b $CUBE_BRANCH
cd ${INSTALL_DIR}/MagicCube
python -m pip install -e .
cd ..

cd ${INSTALL_DIR}
export AUTODIST_DEVOPS="https://${DEVOPS_TOKEN}@msrasrg.visualstudio.com/SuperScaler/_git/AutoDist"
git clone $AUTODIST_DEVOPS -b $AUTODIST_BRANCH
cd ${INSTALL_DIR}/AutoDist
python -m pip install -e .
cd ..

cd ${INSTALL_DIR}
export FAIRSEQ_DEVOPS="https://${DEVOPS_TOKEN}@msrasrg.visualstudio.com/SuperScaler/_git/Fairseq"
git clone $FAIRSEQ_DEVOPS -b $FAIRSEQ_BRANCH
cd ${INSTALL_DIR}/Fairseq
python -m pip install -e .

python -c 'import torch; import cube; import fairseq; print(torch.__path__, cube.__path__, fairseq.__path__)'
