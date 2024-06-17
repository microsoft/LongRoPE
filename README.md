# LongRoPE

**LongRoPE** is a project to extend the context window of pre-trained LLMs to a maximum of 2048k tokens by modifying positional embeddings.

Here is the link of our paper: [LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens](https://arxiv.org/pdf/2402.13753)

## What does LongRoPE do?

The LongRoPE algorithm is inspired by the discovery of the two forms of non-uniformities in positional inter polation: varying RoPE dimensions and token positions. In order to achieve the best performance on long context windows using non-uniform positional embeddings, LongRoPE:
- Exploit the best positional embedding rescaling parameters through an efficient search, providing a better initialization for fine-tuning and enabling an 8x extension in non-fine-tuning scenarios;
- Introduce a progressive extension strategy that first fine-tunes a 256k length LLM and then con ducts a second positional interpolation on the fine tuned extended LLM to achieve a 2048k context window;
- Readjust scaling factors and retained start tokens on 8k length to recover the short context window performance.

## What is LongRoPE’s intended uses?

LongRoPE intent to search for optimal RoPE rescale factors to extent context window of LLMs. Users could apply our code to get longer context window of their own models.

Intended audience for this release should be researchers who want to extend context window length of their own models. To use this code safely and appropriately, users should carefully read our [paper](https://arxiv.org/pdf/2402.13753) first.

> Note: Additional validation would need to be done before this was used in production environments. This is not intended for production use.

> Note: LongRoPE’s code currently only supports English.

## LongRoPE Performance

We evaluate LongRoPE on following metrics:

### Long-Context Perplexity

- **Proof-pile**

| Context Window | 4096 | 8192 | 32768 | 65536 | 98304 | 131072 | 262244 |
| :-------------: | :----------------: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: |
| LongRoPE-LLaMA2-2048k | 3.85 | 3.65 | 3.63 | 2.38 | 2.28 | 2.26 | 1.87 |
| LongRoPE-Mistral-2048k | 3.20 | 3.04 | 2.36 | 2.18 | 2.13 | 2.14 | 1.84 |

- **Books3**

| Context Window | 8k | 16k | 32k | 64k | 128k | 256k | 512k | 1024k | 2048k |
| :-------------: | :----------------: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: |
| LongRoPE-LLaMA2-2048k | 6.81 | 6.66 | 6.31 | 6.27 | 6.21 | 6.17 | 6.17 | 6.35 | 7.08 |
| LongRoPE-Mistral-2048k | 6.63 | 6.48 | 6.38 | 6.43 | 6.68 | 7.15 | 7.98 | 9.42 | 13.71 |

- **PG19**

| Context Window | 8k | 64k | 128k |
| :-------------: | :-----: | :-----: | :-----: |
| LongRoPE-LLaMA2-2048k | 7.37 | 6.64 | 6.31 |
| LongRoPE-Mistral-2048k | 7.10 | 6.98 | 7.13 |

### HuggingFace Open LLM Benchmark

| | Context Window | ARC-C | HellaSwag | MMLU | TruthfulQA |
| :-------------: | :------: | :-----: | :-----: | :-----: | :-----: |
| LongRoPE-LLaMA2-2048k | 2048k | 51.0 | 75.3 | 39.6 | 37.3 |
| LongRoPE-Mistral-2048k | 2048k | 59.2 | 80.9 | 61.1 | 42.2 |


### Passkey Accuracy
Measure the retrieval accurary of key information in long garbage context:
| Context Window | 4k, 8k, 16k, 64k, 100k, 128k, 160k, 256k, 512k | 1024k | 1800k | 2048k |
| :-------------: | :------: | :-----: | :-----: | :-----: |
| LongRoPE-LLaMA2-2048k | 100% | 100% | 100% | 60% |
| LongRoPE-Mistral-2048k | 100% | 90% | 90% | 90% |


## Use LongRoPE to Extend Content Window

### Build Environment

``` bash
conda create -n longrope python==3.11
conda activate longrope
# flash-attn needs cuda >= 11.7
pip install -r requirements.txt
```

### Tokenize Data

Tokenize PG19 as evolution search validation dataset and Proof-Pile as evaluation dataset.

```bash
bash ./examples/llama3/tokenzie-data.sh
```

### Evolution Search

Run evoluation search on Llama-3-8B model to sequence length of 128k:

``` bash
bash ./examples/llama3/search.sh
```

The default evolution search hyperparameters are located in `evolution/default_hyper_params/*.json`. Users can customize the number of iterations, population size, number of parents, number of mutation and crossover operations in each iteration. These parameters will affect the convergence time and robustness of searching results.

### Evaluation
Evaluate long-context perplexity and passkey accuracy:
``` bash
bash ./examples/llama2/evaluate.sh
```


## Others

There are some potential limitations of LongRoPE, such as:
- High GPU memory occupation: The LongRoPE pipeline includes long context inference and training which requires more GPU memory for activations and gradients. The users can apply memory-saving technics such as tensor parallel, ZeRO offloading and recomputation.
- Low generation throughput:  The generation throughput may be limited by (1) the high memory requirement to open K / V caches and (2) the long latency of attention operation for a new token with long context window. The users can parallelize, quantize and / or prune K / V caches to accelerate long-context generation.
- As we prepare to release the code for LLMs, there indeed exists the possibility of unintentional misuse. For instance:
  - Misuse by non-professional users: Individuals who are not familiar with the code or are beginners may not fully comprehend how to use it, leading to its inability to function properly and potential misuse.
  - Usage in non-designed environments: We have only tested the code in English environments. If the code is utilized in operational environments for which it was not designed, such as Chinese environments, it could yield unexpected results.

## Citation

If you find that LongRoPE helps your research, please consider citing it:
```
@misc{ding2024longrope,
      title={LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens}, 
      author={Yiran Ding and Li Lyna Zhang and Chengruidong Zhang and Yuanyuan Xu and Ning Shang and Jiahang Xu and Fan Yang and Mao Yang},
      year={2024},
      eprint={2402.13753},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```