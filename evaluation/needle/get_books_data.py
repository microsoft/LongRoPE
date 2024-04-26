import torch
import numpy as np
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_from_disk

teamdrive = "/mnt/yiran/teamdrive3/ExtendSeqLen"
# model_bf16 = teamdrive + "/ft_out_model/cube-mis-256k-bf16-step-500/ck-1_500"
# model_fp16 = teamdrive + "/ft_out_model/cube-16k-mistral-128k/ck-400"

model_fp16 = teamdrive + "/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_fp16, use_fast = True )

books_data = teamdrive + "/books3-test-sampled-1024k-tokenized"

# BOOKS3_256K="--tokenized ${path_team}/books3-test-sampled-1024k-tokenized --dataset_min_tokens 2097152 --samples 20 --sliding_window 262144"
dataset_min_tokens=2097152
samples = 20
input_texts = load_from_disk(books_data)

config = AutoConfig.from_pretrained(model_fp16, cache_dir="/mnt/yiran/cache")
save_path = f"books_type_{config.model_type}_min{dataset_min_tokens}.pt"
if os.path.exists(save_path):
    print("load from pt")
    input_texts = torch.load(save_path)
    print("load finish")
else:
    if dataset_min_tokens:
        input_texts = input_texts.filter(
            lambda x: x["tokenized_len"] >= dataset_min_tokens)
    if samples:
        input_texts = input_texts[:min(samples, len(input_texts))]
    torch.save(input_texts, save_path)

print("sample nums", len(input_texts['input_ids']))

# exit(0)
# with open("text.txt", "w", encoding="utf-8") as file:  # 打开文件以写入，使用utf-8编码  
#     for i in range(len(input_texts['input_ids'])):  
#         input_text = input_texts['input_ids'][i]  # 注意这里是[i]而不是[0]  
#         out_ids = input_text[:100] + input_text[-100:]  
#         text = tokenizer.decode(out_ids, skip_special_tokens=True)  
  
#         file.write(f"{i}: {len(input_text)}\n")  # 写入序号和长度  
#         file.write(text + "\n\n")  # 写入解码的文本并加上换行符  

# input_text = input_texts['input_ids'][7]
# input_ids_tensor = torch.tensor([input_text], dtype=torch.int64)  

# print(input_ids_tensor)

# out_ids = torch.cat((input_ids_tensor[0, :100], input_ids_tensor[0, -100:]))  


# 解码tensor到字符串  
# text = tokenizer.decode(out_ids, skip_special_tokens=True) 
# print(text)

# pt_path = "/mnt/yiran/LongRoPE/evaluation/needle/books_7_mistral.pt"
# torch.save(input_ids_tensor.cpu(), pt_path)
# print("torch save")
# input_ids_tensor_re = torch.load(pt_path)
# out_ids = torch.cat((input_ids_tensor_re[0, :100], input_ids_tensor_re[0, -100:]))  
# text = tokenizer.decode(out_ids, skip_special_tokens=True) 
# print(text)

# for p in [19, 16, 15, 14, 11, 9, 8, 4]:
for p in [19]:
    input_text = input_texts['input_ids'][p]
    input_ids_tensor = torch.tensor([input_text], dtype=torch.int64)  
    print(input_ids_tensor)
    # pt_path = f"/mnt/yiran/LongRoPE/evaluation/needle/books_{p}_mistral.pt"
    # torch.save(input_ids_tensor.cpu(), pt_path)
    pt_path = "evaluation/needle/books_19_llama2.pt"
    torch.save(input_ids_tensor.cpu(), pt_path)
    # out_ids = torch.cat((input_ids_tensor[0, :2000], input_ids_tensor[0, -2000:])) 
    # text = tokenizer.decode(input_ids_tensor, skip_special_tokens=True) 
    # with open("evaluation/needle/books_data/text_books19.txt", "w", encoding="utf-8") as file:
    #     # file.write(f"{p}: {len(input_text)}\n")  # 写入序号和长度  
    #     file.write(text + "\n\n")  # 写入解码的文本并加上换行符  
    
