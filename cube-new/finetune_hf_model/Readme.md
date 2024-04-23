# Finetune huggingface model with Cube

## Setup environment

Download the environment setup script from https://msrasrg.visualstudio.com/SuperScaler/_git/Fairseq?path=/cube_scripts/set_env.sh.

Please first setup the environment with `bash set_env.sh <REPO_TOKEN> <INSTALL_DIR> <TORCHSCALE_BRANCH> <CUBE_BRANCH> <FAIRSEQ_BRANCH> <AutoDist_BRANCH> <ENV_OPTION(optional)> <NEW_ENV_NAME(optional)> <CONDA_BASE_ENV(optional)>`, if you don't have a token, please contect with cube group.

Here is an example to setup environment under $HOME folder, which will create a new conda environment cube_ft based on the existed conda environment <CONDA_BASE_ENV>:

```
bash set_env.sh <REPO_TOKEN> $HOME devmain main devmain main conda cube_ft <CONDA_BASE_ENV>
```

After setup cube base environment, additional requirements should be installed:

```
conda activate cube_ft
pip install -r $HOME/Fairseq/cube_examples/finetune_hf_model/requirements.txt
```

## Quick start

If you are new to use Cube, please follow the `./Quickstart.md` to make a quick exploration.

## Variable in run script template

### `torchrun` related

* `NPROC_PER_NODE`: how many GPUs in each node, i.e., if you have 8 GPUs in one node, `NPROC_PER_NODE=8`.
* `NNODES`: how many nodes in your environment, i.e., if you have 2 nodes, `NNODES=2`.
* `NODE_RANK`: the node rank of current node, i.e., on your first node, `NODE_RANK=0`, on your second node, `NODE_RANK=1`.
* `MASTER_ADDR`: `master_addr` used by torchrun.
* `MASTER_PORT`: `master_port` used by torchrun.

### Cube related

* `CUBE_SCALING_FACTOR`: the number of scale unit. Cube divides all GPUs into equal parts, and each part is called a scale unit. Data parallel is applied between scale unit, and apply the same parallel plan inner scale unit.
* `PLAN_NGPUS`: how many GPUs in a scale unit. Note that `$PLAN_NGPUS * $CUBE_SCALING_FACTOR` should equal to `$NPROC_PER_NODE * $NNODES`.
* `USE_ZERO`: `1` means using ZERO stage 1, `0` means don't using ZERO.
* `ZERO_N_GROUPS`: the number of zero groups, the GPUs used in each group is `$NPROC_PER_NODE * $NNODES / $ZERO_N_GROUPS`. The default value is `$NNODES`, means using ZERO in each node. If you want to save more GPU memory, you could reduce this value, which means ZERO across nodes, but be aware that it will introduce more communication. If you want to save communication, than you can use a larger `ZERO_N_GROUPS`, but more GPU memory is used.
* `RECOMPUTE_MODULE_NAMES`: the module types that wants to apply recompute to save GPU memory. The value is a string separated by `,`, for example, `TypeA,TypeB,TypeC`.
* `MEM_PER_GPU`: how much GPU memory (GB) is assumed to be used for training by each GPU. Since usually not all the GPU memory is used to load tensors used in the training process, so this value should be set smaller than the maximum GPU memory. The default value is `0.8 * max_memory_per_gpu`. If you deal will OOM during training, you could reduce this value and `bash run.sh compile` again.

### Path of example, data, log and checkpoint

* `EXAMPLE_PATH`: the folder path where the finetuning example is, the default path is `$HOME/Fairseq/cube_examples/finetune_hf_model`.
* `SCRIPT_PATH`: the folder path where has the `run.sh`.
* `DATASET_NAME`: the prefix of three data file name, we assume you have three data file named `$DATASET_NAME_train.json`, `$DATASET_NAME_test.json`, `$DATASET_NAME_valid.json`
* `DATA_DIR`: the data folder path of the three json files.
* `OUTPUT_DIR`: the output folder of cube, by default is `$SCRIPT_PATH`
* `TENSORBOARD_DIR`: the folder where to save tensorboard events files, by default is `$OUTPUT_DIR/tensorboard`.
* `LOG_DIR`: the folder where to save cube log files, by default is `$OUTPUT_DIR/log`.
* `CHECKPOINT_DIR`: the folder where to save checkpoints, by default is `$OUTPUT_DIR/checkpoint`.

### Model related

* `MODEL_ARCH`: current supported are `hf_auto_lm` (AutoModel), `hf_auto_causal_lm` (AutoModelForCausalLM), `hf_auto_seq2seq_lm` (AutoModelForSeq2SeqLM), `hf_auto_seq_cls_lm` (AutoModelForSequenceClassification), `hf_auto_token_cls_lm` (AutoModelForTokenClassification), `hf_auto_qa_lm` (AutoModelForQuestionAnswering).
* `HUGGINGFACE_MODEL_NAME_OR_PATH`: the huggingface model id or the huggingface model checkpoint floder, for example, the model id of Llama2-13B huggingface format model `meta-llama2/Llama-2-13b-hf`.
* `USE_LORA`: if enable LoRA finetuning.
* `LORA_CONFIG`: config the LoRA, the sub arguments is align with PEFT LoRA https://huggingface.co/docs/peft/package_reference/lora.

### Data related

More information please view `./CustomizeDataloader.md`.

* `DATASET_TYPE`: `instruction` and `disk` is supported.
* `DATASET_PATH`: where the data saved.
* `PROMPT_TYPE`: used by `instruction` type. default is `alpaca`.

### Finetuning related

* `SEQUENCE_LENGTH`: the sequence length of the input.
* `BATCH_SIZE`: batch size in one scale unit, the global batch size is `$BATCH_SIZE * $CUBE_SCALING_FACTOR`.
* `GRADIENT_ACCU_NUM`: gradient accumulation times, how many batches to apply a step.
* `WARMUP_STEPS`: integer, warmup steps.
* `MAX_STEPS`: integer, max training steps.
* `SEED`: integer, random seed for finetuning.
* `OPTIMIZE_PARAMS`: optimizer related setting.

## Advance usage

The following use cases are shown in `./Advance.md`.

- Apply recompute to save GPU memory
- Apply pipeline parallel to optimize distributed training
- Apply LoRA training
- Apply flash attention

## Customize

- customized function
- customized model
- customized dataloader
- customized criterion (loss function)
