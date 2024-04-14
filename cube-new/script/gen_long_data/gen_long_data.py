
import datasets  
from datasets import load_dataset, Dataset  
from tqdm import tqdm  
from transformers import AutoTokenizer  
# /mnt/yiran/cache/RedPajama-128k
from pathlib import Path
import torch

# data = datasets.load_from_disk("/mnt/yiran/cache/RedPajama-128k")  
data = datasets.load_from_disk("/mnt/yiran/cache/Long-Data-Collections-mistral-16k")  
  
print(data)
# exit(0)
# 计算可以整除2的行数  
scale = 4
num_full_batches = len(data) // scale
print("data", data)
print("len, ", len(data), num_full_batches)
# num_full_batches = 100


# 初始化字典以存储拼接后的数据  
concatenated_data = {  
    'input_ids': [],  
    'attention_mask': [],  
    'labels': [],
}  

## size
sizes = []

# # 使用tqdm来拼接数据并显示进度  
for i in tqdm(range(0, num_full_batches * scale, scale), desc='Concatenating data'):  
    # 拼接记录  
    concatenated_data['input_ids'].append(sum([data[i+j]['input_ids'] for j in range(scale)], []))  
    print("$", len(concatenated_data['input_ids'][0]), \
        concatenated_data['input_ids'][i//scale][:5], \
        concatenated_data['input_ids'][i//scale][-5:])
    concatenated_data['attention_mask'].append(sum([data[i+j]['attention_mask'] for j in range(scale)], []))  
    concatenated_data['labels'].append(sum([data[i+j]['labels'] for j in range(scale)], []))  

    sizes.append(len(concatenated_data['input_ids']))
    # print("concatenated_data attention_mask", concatenated_data['attention_mask'])

print("$len", len(concatenated_data['input_ids'][0]))
# 转换为`Dataset`对象  
concatenated_dataset = Dataset.from_dict({  
    'input_ids': concatenated_data['input_ids'],  
    'attention_mask': concatenated_data['attention_mask'], 
    'labels': concatenated_data['labels'],
})  
  

# # 保存数据到磁盘  
mapped_save_path = "/mnt/yiran/cache/Long-Data-Collections-mistral-64k" 

sizes = np.array(sizes)
torch.save(sizes, Path(mapped_save_path) / 'sizes.np')
print(concatenated_dataset)

concatenated_dataset.save_to_disk(mapped_save_path) 


