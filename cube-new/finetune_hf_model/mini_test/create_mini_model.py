import argparse
from transformers import AutoTokenizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('tokenizer_id')
    parser.add_argument('save_dir')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id)

    from transformers import LlamaConfig, LlamaForCausalLM
    config_mini = {
        "architectures": [
            "LlamaForCausalLM"
        ],
        "bos_token_id": 1,
        "eos_token_id": 2,
        "hidden_act": "silu",
        "hidden_size": 768,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "max_position_embeddings": 2048,
        "model_type": "llama",
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "num_key_value_heads": 12,
        "pretraining_tp": 1,
        "rms_norm_eps": 1e-05,
        "rope_scaling": None,
        "tie_word_embeddings": False,
        "torch_dtype": "float16",
        "use_cache": True,
        "vocab_size": len(tokenizer)
        }

    model = LlamaForCausalLM(LlamaConfig(**config_mini))
    model.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)
