import os 
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import glob
# teamdrive = "/mnt/yiran/teamdrive/ExtendSeqLen"
teamdrive = "/mnt/yiran/teamdrive3/ExtendSeqLen"
# model_bf16 = teamdrive + "/ft_out_model/cube-mis-256k-bf16-step-500/ck-1_500"
# model_fp16 = teamdrive + "/ft_out_model/cube-16k-mistral-128k/ck-400"
model_fp16 = teamdrive + "/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_fp16, use_fast = True )

data_path = "evaluation/needle/PaulGrahamEssays/"
# input_text = "today is sunny day."
cnt = 0
cnt_nums = 0
for file in glob.glob(f"{data_path}*.txt"):
    # if cnt > 1:
    #     break
    # else:
    #     cnt += 1
    with open(file, 'r') as f:
        file_text = f.read()
        file_name = file.split('/')[-1].split('.')[-2]
        
        print(f"read {file_name}, \n  str len {len(file_text)}")
        input_ids = tokenizer.encode(file_text, return_tensors="pt")
        print(f"input_ids.shape {input_ids.shape}")
        print(input_ids.dtype)
        cnt += input_ids.shape[1]
        cnt_nums += 1
        torch.save(input_ids.cpu(), f'{data_path}{file_name}_la2.pt')
        # 164109 
print("cnt", cnt)
print("cnt_nums", cnt_nums)

#
# cnt 164109 tokens * 12 => 2M 
# cnt_nums 49


# reload_f = torch.load("/mnt/yiran/LongRoPE/evaluation/needle/PaulGrahamEssays/submarine.pt") 
# print(reload_f.shape)
# print(reload_f[0][-5:])
# print(tokenizer.decode(reload_f[0][-5:], skip_special_tokens=True))
# input_ids_encode = tokenizer.encode(input_text, return_tensors="pt")
# print("input_ids_encode", input_ids_encode)

# input_ids_tokenizer = tokenizer(input_text, return_tensors="pt")
# print("input_ids_tokenizer", input_ids_tokenizer)

# output_text_encode = tokenizer.decode(input_ids_encode[0][:], skip_special_tokens=True)
# print("\noutput_text_encode", output_text_encode)