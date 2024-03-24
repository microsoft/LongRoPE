# LongRoPE

## Build Environment
- conda: needs flash-attn (cuda >= 11.7)
  - `conda create -n longrope python==3.11`
  - `conda activate longrope`
  - `cd s-PI`
  - `pip install -r requirements.txt`
  - ``

## Eval:

### PPL
- `cd s-PI`
  - `bash ./script/ppl_eval/ppl_eval_llama_2.sh`

### Passkey
- `cd s-PI`
  - `bash ./script/pk_eval/pk_test_ft_la2.sh`

### HF Eval
- `conda create -n longrope_hf_eval python==3.10.13`
- `conda activate longrope_hf_eval`
- `cd evaluation/lm-evaluation-harness/`
- `pip install -e .`
- `cd ../../`
- `pip install -r requiements.txt`


### Search:
#### Search the scale for base(Llama2-7b 4k) to 256k sequences
- `cd s-PI`
  - `bash ./script/ppl_search/ppl_search_dim_mono_256k.sh`

#### Search the scale for LongRoPE-256k to 512k sequences
- `cd s-PI`
  - `bash ./script/ppl_search/ppl_search_dim_mono_512k-la2-256k.sh`




