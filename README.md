# LongRoPE2: Near-Lossless LLM Context Window Scaling

`LongRoPE2` is a novel approach that extends the effective context window of pre-trained LLMs to a target length (e.g., 128k) while **preserving the model's original performance on short-context tasks**.

While our previous work, [LongRoPE](https://github.com/microsoft/LongRoPE), successfully extended context windows to 2048k, it faced a common challenge among extension methods: a noticeable performance degradation on standard short-context benchmarks.

`LongRoPE2` directly solves this "near-lossless" challenge through three key contributions:

1.  **A new hypothesis** on RoPE OOD issues, identifying insufficient training in higher RoPE dimensions as the root cause.
2.  **An evolutionary search algorithm** guided by "needle-driven" perplexity to find the *true* critical RoPE dimensions and optimal rescaling factors.
3.  **A mixed context window training approach** that simultaneously fine-tunes the model with original RoPE for short sequences and the rescaled RoPE for long sequences, preserving performance across all context lengths.

Remarkably, `LongRoPE2` extends LLaMA3-8B to an effective 128K context length while **retaining over 98.5% of its original short-context performance**. This is achieved using only 10B training tokens—**80x fewer** than Meta's LLaMA3.1 approach.

*LongRoPE2-extended LLaMA3-8B achieves the best performance at a 128k context length among comparable models.*

## Key Features

  * **Near-Lossless Short Context Performance**: Retains \>98.5% of the original model's performance on standard short-context benchmarks (e.g., MMLU, GSM8K).
  * **Superior Long Context Performance**: Achieves state-of-the-art results on long-context benchmarks like RULER and achieves near-perfect retrieval on "Needle in a Haystack" tests up to 128k.
  * **Extreme Training Efficiency**: Requires only 10B tokens for fine-tuning, 80x fewer than comparable models like LLaMA3.1-8B.
  * **Solves RoPE OOD Issues**: Identifies and corrects for undertraining in high-frequency RoPE dimensions, leading to a more robust extension.
  * **Flexible Inference**: Uses the original RoPE for short-context inputs and automatically switches to the rescaled RoPE for long-context inputs, ensuring optimal performance for any sequence length.

## News

  * **[2025/02/27]** Our paper, "LongRoPE2: Near-Lossless LLM Context Window Scaling," is now available on [arXiv](https://arxiv.org/abs/2502.20082).

-----

## Installation

1.  Clone this repository:

    ```bash
    git clone https://github.com/microsoft/LongRoPE -b longrope2
    cd LongRoPE
    ```

2.  Create and activate a Conda environment:

    ```bash
    conda create -n longrope2 python=3.10
    conda activate longrope2
    ```

3.  Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

The `LongRoPE2` method consists of two main stages:

1.  **RoPE Rescaling Factor Search**: Using evolutionary search guided by "needle-driven" PPL to find the optimal factors.
2.  **Mixed Context Window Training**: Fine-tuning the model to use both original and rescaled RoPE factors.

### RoPE Rescaling Factor Search

Please note that this branch is a dev branch, please view the following scripts for more information.

The needle data generation example for llama3: [pg19_needle_llama3.py](https://github.com/microsoft/LongRoPE/blob/longrope2/examples/longrope2/pg19_needle_llama3.py)
The evolutionary search example for llama3: [search-llama3-long-factor-cd-33-around-init.sh](https://github.com/microsoft/LongRoPE/blob/longrope2/examples/longrope2/search-llama3-long-factor-cd-33-around-init.sh)

*This step generates the scaling factors, which minimize PPL on synthetic "needle-driven" data.*

### Mixed Context Window Training

For the training part, please view [nnscaler-longrope2-example](https://github.com/microsoft/nnscaler/tree/main/examples/longrope2).

## Results

### Long Context Performance

`LongRoPE2` significantly outperforms prior SOTA methods (NTK, YaRN, and LongRoPE 1.0) on the RULER benchmark, especially at the 128k limit.

**RULER Benchmark (Average Score)**
| Method (Base: LLaMA3-8B) | 4k | 8k | 16k | 32k | 64k | 128k |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| YaRN | 91.86 | 87.87 | 84.67 | 68.80 | 62.51 | 49.39 |
| NTK | 94.38 | 92.64 | 87.33 | 91.93 | 79.26 | 73.19 |
| LongRoPE (Gen 1) | 94.60 | 92.70 | 86.60 | 91.01 | 81.23 | 73.40 |
| **LongRoPE2 (Ours)** | **94.61** | **93.68** | **92.31** | **90.49** | **85.62** | **82.03** |

`LongRoPE2` also achieves near-perfect, 100% retrieval accuracy on the "Needle in a Haystack" test across all depths and context lengths up to 128k.

### Short Context Performance (Near-Lossless)

The primary goal of `LongRoPE2` is to prevent short-context performance degradation. The model retains over 98.6% of its original capabilities, solving the key drawback of previous methods.

**Standard Short-Context Benchmarks (LLaMA3-8B Base)**
| Model (LLaMA3-8B) | Avg. | MMLU | MMLU-Pro | HellaSwag | TruthfulQA | GSM8K |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Original LLaMA3-8B (8k) | 56.5 | 66.62 | 35.87 | 82.08 | 44.04 | 54.05 |
| YaRN | 52.1 | 62.25 | 31.88 | 81.25 | 42.61 | 42.45 |
| NTK | 54.0 | 63.84 | 34.14 | 82.11 | 43.45 | 46.92 |
| LongRoPE (Gen 1) | 54.6 | 64.69 | 33.74 | **82.14** | 43.65 | 48.90 |
| **LongRoPE2 (Ours)** | **55.7** | **65.01** | **34.61** | 81.69 | **46.17** | **50.80** |
| *% Retained (vs. Original)* | *98.6%* | *97.6%* | *96.5%* | *99.5%* | *104.8%* | *94.0%* |

## Citation

If you find `LongRoPE2` useful in your research, please cite our paper:

```bibtex
@misc{shang2025longrope2nearlosslessllmcontext,
      title={LongRoPE2: Near-Lossless LLM Context Window Scaling}, 
      author={Ning Shang and Li Lyna Zhang and Siyuan Wang and Gaokai Zhang and Gilsinia Lopez and Fan Yang and Weizhu Chen and Mao Yang},
      year={2025},
      eprint={2502.20082},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.20082}, 
}
```

## Acknowledgements

  * Our work builds upon our previous project, [microsoft/LongRoPE](https://github.com/microsoft/LongRoPE).
  * We utilized [nnScaler](https://github.com/microsoft/nnscaler) for efficient distributed training.
