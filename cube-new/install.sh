# pip install datasets
# pip install numpy
# pip install pandas
# pip install einops
# pip install tqdm
# pip install regex
# pip install git+https://github.com/shumingma/infinibatch.git
# pip install iopath
# pip install boto3
# pip install dill
# pip install scipy
# pip install sentencepiece
# pip install more_itertools
# pip install numpy==1.23.0
# pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
# pip install tensorboard
# pip install psutil
# pip install transformers==4.36.2
# pip install peft
# pip install fairscale==0.4.0
# pip install hydra-core==1.0.7 omegaconf==2.0.6
# pip install git+https://github.com/shumingma/infinibatch.git
# pip install flash-attn --no-build-isolation
# pip install accelerate==0.25.0

# git clone -b 22.04-dev https://github.com/NVIDIA/apex.git
# cd apex
# pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
# cd ..

# bash set_env.sh \
#     <REPO_TOKEN> \
#     <INSTALL_DIR> \
#     <TORCHSCALE_BRANCH> \
#     <CUBE_BRANCH> \
#     <FAIRSEQ_BRANCH> \
#     <AUTODIST_BRANCH> \
#     <ENV_OPTION(optional)> \
#     <NEW_ENV_NAME(optional)> \
#     <CONDA_BASE_ENV(optional)>

bash set_env.sh \
     \
    /mnt/yiran/cube-new-cz \
    devmain \
    nishang/longseq \
    nishang/longseq \
    yizhu1/long_seq_dev \
    conda \
    cube-nishang \
    nishang



