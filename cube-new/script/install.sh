
# conda install python==3.10.14 
# pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
# pip install transformers==4.36.2
# pip install accelerate==0.25.0
# pip install numpy==1.23.0
# pip install datasets
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

# pip install tensorboard
# pip install psutil

# pip install peft
# pip install fairscale==0.4.0
# pip install hydra-core==1.0.7 omegaconf==2.0.6
# pip install git+https://github.com/shumingma/infinibatch.git
# pip install flash-attn --no-build-isolation


# git clone -b 22.04-dev https://github.com/NVIDIA/apex.git
# change apex/apex/amp/_initialize.py:#L2: "string_classes = str"
# comment apex/setup.py:#L35-#L43

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




