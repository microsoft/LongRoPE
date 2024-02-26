# LongRoPE

## Build Environment
- conda: needs flash-attn (cuda >= 11.7)
  - `conda create -n scale-rope python==3.11`
  - `conda activate scale-rope`
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

###  

### Search:
#### Search the scale for base(Llama2-7b 4k) to 256k sequences
- `cd s-PI`
  - `bash ./script/ppl_search/ppl_search_dim_mono_256k.sh`

#### Search the scale for LongRoPE-256k to 512k sequences
- `cd s-PI`
  - `bash ./script/ppl_search/ppl_search_dim_mono_512k-la2-256k.sh`




