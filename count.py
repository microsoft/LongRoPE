import datasets
import os
os.environ["HF_HOME"]="/mnt/yiran/cache"
from tqdm import tqdm

# # "togethercomputer/RedPajama-Data-1T-Sample",
# name = "RedPajama-Data-1T-Sample"
# name = "Long-Data-Collections-fine-tune"
name = "Long-Data-Collections-pretrain"

# dataset = datasets.load_dataset( path="/mnt/yiran/cache/togethercomputer___red_pajama-data-1_t-sample/plain_text/1.0.0/6ea3bc8ec2e84ec6d2df1930942e9028ace8c5b9d9143823cf911c50bbd92039", cache_dir = "/mnt/yiran/cache/", split="train")
# dataset = datasets.load_dataset("/mnt/yiran/cache/Long-Data-Collections", "pretrain", cache_dir = "/mnt/yiran/cache/", split="train")
dataset = datasets.load_dataset("/mnt/yiran/cache/Long-Data-Collections/", data_files="/mnt/yiran/cache/long/P3_decontaminated_materialized.jsonl.zst", cache_dir = "/mnt/yiran/cache/", split="train")
print(dataset)

len_dict = {"0-4096": 0, 
            "4096-8192": 0,
            "8192-16384": 0,
            "16384-32768": 0,
            "32768-65536": 0,
            "65536-131072": 0,
            "131072-262144": 0,
            "262144+":0,
            }
for i in tqdm(range(dataset.num_rows)):
    str_len = len(dataset[i]['text'])
    token_len = str_len // 4
    if token_len < 4096:  
        len_dict["0-4096"] += 1  
    elif token_len < 8192:  
        len_dict["4096-8192"] += 1  
    elif token_len < 16384:  
        len_dict["8192-16384"] += 1  
    elif token_len < 32768:  
        len_dict["16384-32768"] += 1  
    elif token_len < 65536:  
        len_dict["32768-65536"] += 1  
    elif token_len < 131072:  
        len_dict["65536-131072"] += 1  
    elif token_len < 262144:  
        len_dict["131072-262144"] += 1  
    else:  
        len_dict["262144+"] += 1  
        
print(len_dict)

# len_dict = {'0-4096': 880966, '4096-8192': 30274, '8192-16384': 13168, '16384-32768': 4485, '32768-65536': 1079, '65536-131072': 373, '131072-262144': 155, '262144+': 14}\
    
NI_decontaminated_materialized = {'0-4096': 39323, '4096-8192': 331695, '8192-16384': 205930, '16384-32768': 725, '32768-65536': 310, '65536-131072': 60, '131072-262144': 0, '262144+': 0}

pile_sub = {'0-4096': 1808968, '4096-8192': 68044, '8192-16384': 46322, '16384-32768': 12116, '32768-65536': 2047, '65536-131072': 1544, '131072-262144': 1150, '262144+': 267}

P3_decontaminated_materialized = {'0-4096': 0, '4096-8192': 2969, '8192-16384': 811017, '16384-32768': 29, '32768/s]-65536': 0, '65536-131072': 0, '131072-262144': 0, '262144+': 0}  

rp_sub = {'0-4096': 880966, '4096-8192': 30274, '8192-16384': 13168, '16384-32768': 4485, '32768-65536': 1079, '65536-131072': 373, '131072-262144': 155, '262144+': 14}

import matplotlib.pyplot as plt  
  

# 创建条形图  
plt.figure(figsize=(10, 6), dpi=400)  
plt.bar(len_dict.keys(), len_dict.values(), color='skyblue')  
  
plt.title('Text Length Distribution')  
plt.xlabel('Token length ranges')  
plt.ylabel('Number of samples')  
  
# 在条形图上显示数值  
for i, (key, value) in enumerate(len_dict.items()):  
    value_p = value / dataset.num_rows * 100  # 计算百分比  
    percentage = "{:d}, {:.4f}%".format(value, value_p)  # 格式化为保留两位小数的百分比字符串  
    plt.text(i, value, percentage, ha='center', va='bottom', fontsize=8)  
  
# 显示图表  
plt.tight_layout()  
plt.savefig(f"img-{name}.png")
plt.show()  


