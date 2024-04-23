# Quick start with a mini-llama & alpaca dataset

It is recommanded to create a experiment folder to continue. We assume the Fairseq folder is under home in this quick start. 

```
export ROOT_PATH=$HOME
export EXAMPLE_PATH=$ROOT_PATH/Fairseq/cube_examples/finetune_hf_model
export EXP_PATH=$ROOT_PATH/quick_start_finetune_llama_mini

mkdir $EXP_PATH
cd $EXP_PATH
```

All scripts you need is under `$EXAMPLE_PATH/mini_test`.

## Create a mini-llama model

You could create a mini llama model checkpoint by:

```
python create_mini_model.py <tokenizer-name-or-path> <save-folder>
```

For example, reuse the llama-2-7b tokenizer, and save the mini model in `$EXP_PATH/mini-llama`.
If you don't have the permission to access https://huggingface.co/meta-llama/Llama-2-7b-hf, using any other tokenizer is ok.

```
python $EXAMPLE_PATH/mini_test/create_mini_model.py meta-llama/Llama-2-7b-hf $EXP_PATH/mini-llama
```

## Finetune the mini-llama with alpaca dataset

At first, we copy the run script and data to the experiment folder.

```
cp $EXAMPLE_PATH/mini_test/run_mini.sh $EXP_PATH
cp $EXAMPLE_PATH/mini_test/alpaca_data_100.json $EXP_PATH
```

The key `HUGGINGFACE_MODEL_NAME_OR_PATH` in the `run_mini.sh` should be filled in, it is the name or path of the huggingface model. You can directly write a huggingface model hub id like `meta-llama/Llama-2-7b-hf` or a local huggingface model checkpoint which can be load by transformers `from_pretrained` API.

In this quick start, we fill in the previous saved mini-llama checkpoint path `$EXP_PATH/mini-llama`.

```
HUGGINGFACE_MODEL_NAME_OR_PATH=$EXP_PATH/mini-llama
```

The entire finetuning (training) process with Cube can be roughly divided into three stages: trace, compile and run.

### Trace

`trace` dumps a pytorch model defined by user to a dataflow graph, model parameters and buffers (train from scratch or load from existing checkpoints).

Execute the following command:

```
# log can be found in ./log/trace_log.txt
bash run_mini.sh trace
```

You can find `fullmodel.pt.*`, `dist_map.pt`, `graph.cube` under `$EXP_PATH/.cube/finetuning` after trace.
* `fullmodel.pt.*` is the saved parameters & buffers from the original model.
* `dist_param_map.pt` is the mapping of weight name used in cube model across the original model. 
* `graph.cube` is the cube model graph traced from the original model.

### Compile

`compile` search a distribution plan for a given model graph and generate the excutable pytorch code.

```
# log can be found in ./log/compile_log.txt
bash run_mini.sh compile
```

You can find `gencode*.py` under `$EXP_PATH/.cube/finetuning` after compile.
The number of gencode files equals to the total GPUs used for training, and each GPU will load model from the corresponding file according to the rank.

### Run

`run` will do training with the distributed model in a `CubeTrainer`. `CubeTrainer` is a customized fairseq trainer to train cube model.

```
# log can be found in ./log/run_log.txt
bash run_mini.sh run
```

Tensorboard events are saved in `$EXP_PATH/tensorboard` folder. Checkpoint files are saved in `$EXP_PATH/checkpoint`.

You could view the metric during training via:

```
tensorboard --logdir $EXP_PATH/tensorboard
```

### Checkpoint

The checkpoint saved by cube is sharded, so we need to merge them to a full one before we load it by huggingface transformers.

```
bash run_mini.sh mergeckpt $EXP_PATH/checkpoint/checkpoint_last-shard0.pt
bash run_mini.sh extract_model_state $EXP_PATH/checkpoint/checkpoint_last-full.pt
```

`last` in `checkpoint_last-shard0.pt` means this is the last checkpoint, `shard0` means this checkpoint is from rank 0. Only given checkpoint on rank 0 is enough, Cube will auto search the remaining shard files under the same folder.

After `mergeckpt`, you could find the `checkpoint_last-full.pt` under checkpoint folder, `checkpoint_last-full.pt` contains all states during training, i.e., model states, optimizer states, dataloader states and so on.

After `extract_model_state`, you could get a pure model state `pytorch_model.bin` under checkpoint folder, then you can take it to do any thing you want.

### Resume

If training is interrupted unexpectedly, the experiment can be easily resumed by:

```
# for a checkpoint file name 'checkpoint_last-0',
# the prefix is 'checkpoint_last',
# the suffix is '-0'
# Fairseq only need prefix and will auto append suffix on each rank.

bash run_mini.sh resume <checkpoint_prefix>
```

Model, optimizer and dataloader states will recover from `<checkpoint_prefix><checkpoint_suffix>.pt`.
