
from datasets import load_dataset, load_from_disk
import transformers


path = "/mnt/yiran/cache/RedPajama-Data-1T-Sample"
model_path = "/mnt/yiran/Llama-2-7b-hf"
num_proc = 4

data = load_dataset(path, split='train')
print(data)
# data = data.select(range(40))

IGNORE_IDX = -100

def get_tokenizer(tokenizer_name_or_path,
                  model_max_length=None,
                  default_bos_token="<s>",
                  default_eos_token="</s>",
                  default_pad_token="[PAD]",
                  default_unk_token="<unk>"):

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        model_max_length=model_max_length,
        padding_side="right",
        trust_remote_code=True,
        # use_fast=False,
    )

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = default_pad_token
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = default_eos_token
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = default_bos_token
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = default_unk_token

    tokenizer.add_special_tokens(special_tokens_dict)
    if model_max_length:
        tokenizer.model_max_length = model_max_length
    return tokenizer



def tokenize_fn(sample, tokenizer):
    text = sample['text']
    inputs = tokenizer(
        text, 
        padding=False, 
        truncation=False,
    )
    sample['input_ids'] = inputs['input_ids']
    return sample
    
    
tokenizer = get_tokenizer(model_path)
data_token = data.map(
    tokenize_fn,
    num_proc=num_proc,
    remove_columns=["text", "meta"],
    desc=" tokenize_fn", 
    fn_kwargs={"tokenizer": tokenizer}
)
print(data_token)
print(len(data_token[0]['input_ids']))

def filter_short_func(x):
    return len(x['input_ids']) < 4096

print("fliter short")
short_dataset = data_token.filter(filter_short_func, desc="fliter short", num_proc=num_proc)
print(short_dataset)

short_dataset.save_to_disk("/mnt/yiran/RedPajama-Data-1T-Sample-less-4k-tokenized")

def filter_long_func(x):
    return len(x['input_ids']) >= 4096

print("fliter long")
long_dataset = data_token.filter(filter_long_func, desc="fliter long", num_proc=num_proc)
print(long_dataset)

long_dataset.save_to_disk("/mnt/yiran/RedPajama-Data-1T-Sample-longer-4k-tokenized")