import datasets
from datasets import load_dataset, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from multiprocessing import Pool, cpu_count
import numpy as np
from pathlib import Path
import torch
import time
import argparse

s_t = time.time()
# data = datasets.load_from_disk("/mnt/yiran/cache/RedPajama-128k")
# data = datasets.load_from_disk("/mnt/yiran/cache/Long-Data-Collections-mistral-16k")  
# mapped_save_path = "/mnt/yiran/cache/Long-Data-Collections-mistral-64k"
# scale = 4


parser = argparse.ArgumentParser()
parser.add_argument('--source_data', type=str, required=True)
parser.add_argument('--mapped_save_path', type=str, required=True)
parser.add_argument('--scale', type=int, required=True)
args = parser.parse_args()

data = datasets.load_from_disk(args.source_data)
mapped_save_path = args.mapped_save_path
scale = args.scale

print(data)
if 'RedPajama' in args.source_data:
    model_type = "llama2"
elif 'Long-Data-Collections' in args.source_data:
    model_type = "mistral"
else:
    raise ValueError("model_type not supports")
    
num_full_batches = len(data) // scale
print("data", data)
print("len, ", len(data), num_full_batches)

if model_type == 'llama2':
    concatenated_data = {
        'input_ids': [],
        'attention_mask': [],
    }
elif model_type == "mistral":
    concatenated_data = {
        'input_ids': [],
        'attention_mask': [],
        'labels': [],
    }
    
sizes = []

def process_batch(i):
    batch_input_ids = sum([data[i+j]['input_ids'] for j in range(scale)], [])
    batch_attention_mask = sum([data[i+j]['attention_mask'] for j in range(scale)], [])
    if model_type == 'mistral':
        batch_attention_labels = sum([data[i+j]['labels'] for j in range(scale)], [])
        return batch_input_ids, batch_attention_mask, batch_attention_labels
    else:
        return batch_input_ids, batch_attention_mask,

if __name__ == '__main__':
    num_processes = int(cpu_count() *0.2)
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(process_batch, range(0, num_full_batches * scale, scale)), desc='Concatenating data', total=num_full_batches))
    
    for result in results:
        if model_type == 'mistral':
            batch_input_ids, batch_attention_mask, batch_attention_labels = result
        else:
            batch_input_ids, batch_attention_mask = result
            
        concatenated_data['input_ids'].append(batch_input_ids)
        concatenated_data['attention_mask'].append(batch_attention_mask)
        if model_type == 'mistral':
            concatenated_data['labels'].append(batch_attention_labels)
        sizes.append(len(batch_input_ids))

print("$len", len(concatenated_data['input_ids'][0]))

if model_type == 'mistral':
    concatenated_dataset = Dataset.from_dict({
        'input_ids': concatenated_data['input_ids'],
        'attention_mask': concatenated_data['attention_mask'],
        'labels': concatenated_data['labels'],
    })
else:
    concatenated_dataset = Dataset.from_dict({
        'input_ids': concatenated_data['input_ids'],
        'attention_mask': concatenated_data['attention_mask'],
    })

sizes = np.array(sizes)
torch.save(sizes, Path(mapped_save_path) / 'sizes.np')
print(concatenated_dataset)
[]
concatenated_dataset.save_to_disk(mapped_save_path)

print("Time", time.time()-s_t)