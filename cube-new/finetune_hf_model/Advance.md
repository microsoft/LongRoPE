# Advance usage

The following use cases are on the basis of `./Quickstart.md` context.

## Apply recompute to save GPU memory

Recompute is technique to save GPU memory, in somewhere it is also called gradient checkpointing.
For a recompute block, it does not save the activation inner the block during forward,
but only save the input of the block and recompute the activation during backward.

It is easy to apply layer-wise recompute with cube, just specify the module type before `compile` is enough.

Here is an example to recompute all module with type `LlamaDecoderLayer` in llama.

In `run_mini.sh`, modify the variable `RECOMPUTE_MODULE_NAMES`:

```
# if want to recompute multiple types, use ',' to separated the module types
RECOMPUTE_MODULE_NAMES=LlamaDecoderLayer
```

Then recompile the code:

```
bash run_mini.sh compile
```

In the new `gencode*.py`, you could find recompute block.

```
def recompute(...):
    ...

... = ckpt.checkpoint(recompute, ...)
```

Then you could `run` with the new code to save GPU memory.

## Apply pipeline parallel to optimize distributed training

Will coming soon.

## Apply LoRA

In this example, `LoRA` is applied by `peft`, view doc (https://huggingface.co/docs/peft/index).

In `run_mini.sh`, set the variable `USE_LORA=1` to enable LoRA on the origin model, modify `LORA_CONFIG` to config LoRA:

```
USE_LORA=1

LORA_CONFIG="--use-lora \
--lora-rank 8 \
--lora-alpha 8 \
--lora-dropout 0.0 \
--target-modules q_proj,k_proj,v_proj
"
```

Then trace the model with LoRA:

```
bash run_mini.sh trace
```

In `trace_log.txt`, you could find the following log:

```
| INFO | src.model_helper.common | Create lora model and save under mini-llama_tm_q_proj_k_proj_v_proj_rank_8_alpha_8_dropout_0.0_fifo_False_bias_none

| INFO | fairseq_cli.train | CausalLMFairseqWrapper(
  (hf_model): PeftModelForCausalLM(
    (base_model): LoraModel(
      (model): LlamaForCausalLM(
        (model): LlamaModel(
          (embed_tokens): Embedding(32256, 768)
          (layers): ModuleList(
            (0-11): 12 x LlamaDecoderLayer(
              (self_attn): LlamaAttention(
                (q_proj): lora.Linear(
                  (base_layer): Linear(in_features=768, out_features=768, bias=False)
                  (lora_dropout): ModuleDict(
                    (default): Identity()
                  )
                  (lora_A): ModuleDict(
                    (default): Linear(in_features=768, out_features=8, bias=False)
                  )
                  (lora_B): ModuleDict(
                    (default): Linear(in_features=8, out_features=768, bias=False)
                  )
                  (lora_embedding_A): ParameterDict()
                  (lora_embedding_B): ParameterDict()
                )
                (k_proj): lora.Linear(
                  (base_layer): Linear(in_features=768, out_features=768, bias=False)
                  (lora_dropout): ModuleDict(
                    (default): Identity()
                  )
                  (lora_A): ModuleDict(
                    (default): Linear(in_features=768, out_features=8, bias=False)
                  )
                  (lora_B): ModuleDict(
                    (default): Linear(in_features=8, out_features=768, bias=False)
                  )
                  (lora_embedding_A): ParameterDict()
                  (lora_embedding_B): ParameterDict()
                )
                (v_proj): lora.Linear(
                  (base_layer): Linear(in_features=768, out_features=768, bias=False)
                  (lora_dropout): ModuleDict(
                    (default): Identity()
                  )
                  (lora_A): ModuleDict(
                    (default): Linear(in_features=768, out_features=8, bias=False)
                  )
                  (lora_B): ModuleDict(
                    (default): Linear(in_features=8, out_features=768, bias=False)
                  )
                  (lora_embedding_A): ParameterDict()
                  (lora_embedding_B): ParameterDict()
                )
                (o_proj): Linear(in_features=768, out_features=768, bias=False)
                (rotary_emb): LlamaRotaryEmbedding()
              )
              (mlp): LlamaMLP(
                (gate_proj): Linear(in_features=768, out_features=3072, bias=False)
                (up_proj): Linear(in_features=768, out_features=3072, bias=False)
                (down_proj): Linear(in_features=3072, out_features=768, bias=False)
                (act_fn): SiLUActivation()
              )
              (input_layernorm): LlamaRMSNorm()
              (post_attention_layernorm): LlamaRMSNorm()
            )
          )
          (norm): LlamaRMSNorm()
        )
        (lm_head): Linear(in_features=768, out_features=32256, bias=False)
      )
    )
  )
)
```

The initial LoRA checkpoint is saved under `$EXP_PATH/mini-llama_tm_q_proj_k_proj_v_proj_rank_8_alpha_8_dropout_0.0_fifo_False_bias_none`. The created LoRA model can be loaded by:

```
from transformers import AutoModel
from peft import PeftModel

model = AutoModel.from_pretrained("mini-llama")
lora_model = PeftModel.from_pretrained(model, "mini-llama_tm_q_proj_k_proj_v_proj_rank_8_alpha_8_dropout_0.0_fifo_False_bias_none")
```

## Apply fused kernel

Fused kernel can use in the model after register it. Detail will come soon.
