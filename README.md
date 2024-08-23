# LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens
 
**LongRoPE** is an effective approach that extends LLM context window beyond 2048k tokens by non-uniformly rescaling RoPE positional embeddings. LongRoPE is accepted by ICML 2024 and has been integrated into Microsoft Phi-3. Learn more about the work [LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens](https://arxiv.org/pdf/2402.13753):

<p align="center">
  <img src="assets/logo.png" width="500px">
</p>
<p align="center">
    🤗 <a href="https://huggingface.co/papers/2402.13753">Huggingface Daily Paper</a>
</p>
<p align="center">
    <a href="https://mp.weixin.qq.com/s/4ryyv59ofNOD--RCSdqktQ">Microsoft Research Official Blog</a>
</p>
<p align="center">
    <a href="https://www.microsoft.com/en-us/research/blog/research-focus-week-of-march-18-2024/">Microsoft Research Blog</a>
</p>

## LongRoPE in Phi3-128k LLMs
LongRoPE currently supports the following Phi3-128k LLMs with 128k context window.

- [Phi-3-mini-128k-instruct](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct)
- [Phi-3-small-128k-instruct](https://huggingface.co/microsoft/Phi-3-small-128k-instruct)
- [Phi-3-medium-128k-instruct](https://huggingface.co/microsoft/Phi-3-medium-128k-instruct)
- [Phi-3-vision-128k-instruct](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct)

### [RULER](https://github.com/hsiehjackson/RULER)
| Model | Context Window | 4k | 8k | 16k | 32k | 64k | 128k | Avg |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| Gemini-1.5-pro | 1M | 96.7 | 95.8 | 96 | 95.9 | 95.9 | 94.4 | 95.8 |
| GPT-4-1106-preview | 128k | 96.6 | 96.3 | 95.2 | 93.2 | 87 | 81.2 | 91.6 |
| GradientAI/LLaMA3 (70B) | 1M | 95.2 | 93.4 | 93.4 | 89.4 | 82.6 | 72 | 87.7 |
| **Phi3-mini-128k (3.8B)** | **128k** | **92.3** | **91.2** | **90.8** | **87.7** | **79.8** | **65.3** | **84.5** |
| Mixtral-8x22B | 64k | 95.6 | 94.9 | 93.4 | 90.9 | 84.7 | 31.7 | 81.9 |
| ChatGLM (6B) | 128k | 87.8 | 83.4 | 78.6 | 69.9 | 56.0 | 42.0 | 69.6 |
| LongChat (7B) | 32k | 84.7 | 79.9 | 70.8 | 59.3 | 0 | 0 | 49.1 |

### Long context code understanding ([RepoQA](https://github.com/evalplus/repoqa))
| Model | Context Window | Python | cpp | java | typescript | rust | avg |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| GPT-4o-2024-05-13 | 128k | 95 | 80 | 85 | 96 | 97 | 90.6 |
| Gemini-1.5-pro-latest | 1M | 91 | 81 | 91 | 94 | 96 | 90.6 |
| claude-3-opus-20240229 | 200k | 93 | 83 | 88 | 95 | 94 | 90.6 |
| **Phi3-mini-128k-Instruct** | **128k** | **86** | **64** | **73** | **94** | **71** | **77.6** |
| GPT-4-turbo-2024-04-09 | 128k | 84 | 79 | 75 | 89 | 55 | 76.4 |
| Mixtral-8x22B-Instruct-v0.1 | 64k | 60 | 67 | 74 | 83 | 55 | 67.8 |

###  More short tasks
| Model | MMLU | GSM8K | MedQA | AGIEval | BBH-Hard | HumanEval |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **Phi3-mini-128k-Instruct** | **68.1** | **83.6** | **55.3** | **36.9** | **71.5** | **57.9** |
| Mistral-7B | 61.7 | 46.4 | 49.6 | 35.1 | 57.3 | 28 |
| Gemma 7B | 63.6 | 59.8 | 50 | 42.1 | 59.6 | 34.1 |
| LLaMA3-Instruct-8B | 66.5 | 77.4 | 60.5 | 42 | 51.5 | 60.4 |
| Mixtral 8x7B | 68.4 | 64.7 | 62.2 | 45.2 | 69.7 | 37.8 |

### Multi-modality long context support
| Model | MMMU | MMBench | ScienceQA | MathVista | InterGPS | ChartQA |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **Phi3-vision 128k-instruct** | **40.4** | **80.5** | **90.8** | **44.5** | **38.1** | **81.4** |
| LLaVA 1.6-vicuna-7B | 34.2 | 76.3 | 70.6 | 31.5 | 20.5 | 55.0 |
| QWEN-VL Chat | 39.0 | 75.8 | 67.2 | 29.4 | 22.3 | 50.9 |
| LLaMA3-LLaVA Next-8B | 36.4 | 79.4 | 73.7 | 34.8 | 24.6 | 65.8 |
| Claude-3-Haiku | 40.7 | 62.4 | 72.0 | 33.2 | 32.1 | 59.3 |
| Gemini 1.0 Pro V | 42.0 | 80.0 | 79.7 | 35.0 | 28.6 | 58.0 |
| GPT-4V Turbo | 55.5 | 86.1 | 75.7 | 47.5 | 41.0 | 62.3 |

## What does LongRoPE do?

The LongRoPE algorithm is built upon the two forms of non-uniformities in positional interpolation: varying RoPE dimensions and token positions. In order to achieve the best performance on long context windows using non-uniform positional embeddings, LongRoPE:
- Exploit the best positional embedding rescaling parameters through an efficient search, providing a better initialization for fine-tuning and enabling an 8x extension in non-fine-tuning scenarios;
- Introduce a progressive extension strategy that first fine-tunes a 256k length LLM and then conducts a second positional interpolation on the fine-tuned extended LLM to achieve a 2048k context window;
- Readjust scaling factors and retained start tokens on 8k length to recover the short context window performance.

Due to policy restrictions, only evolution search part is now released. Any LLM training techniques such as [EasyContext](https://github.com/jzhang38/EasyContext) and [nnScaler](https://github.com/microsoft/nnscaler) can be applied to the fine-tuning stage.


## Quick Start

### Build Environment

``` bash
conda create -n longrope python==3.10
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
bash ./examples/llama3/evaluate.sh
```


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
