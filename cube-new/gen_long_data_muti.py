import datasets
from datasets import load_dataset, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from multiprocessing import Pool, cpu_count
import numpy as np
from pathlib import Path
import torch
import time

s_t = time.time()
# data = datasets.load_from_disk("/mnt/yiran/cache/RedPajama-128k")
data = datasets.load_from_disk("/mnt/yiran/cache/Long-Data-Collections-mistral-16k")  
print(data)

scale = 4
num_full_batches = len(data) // scale
print("data", data)
print("len, ", len(data), num_full_batches)

concatenated_data = {
    'input_ids': [],
    'attention_mask': [],
}

sizes = []

def process_batch(i):
    batch_input_ids = sum([data[i+j]['input_ids'] for j in range(scale)], [])
    batch_attention_mask = sum([data[i+j]['attention_mask'] for j in range(scale)], [])
    return batch_input_ids, batch_attention_mask

if __name__ == '__main__':
    num_processes = cpu_count() - 1
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(process_batch, range(0, num_full_batches * scale, scale)), desc='Concatenating data', total=num_full_batches))
    
    for result in results:
        batch_input_ids, batch_attention_mask = result
        concatenated_data['input_ids'].append(batch_input_ids)
        concatenated_data['attention_mask'].append(batch_attention_mask)
        sizes.append(len(batch_input_ids))

print("$len", len(concatenated_data['input_ids'][0]))

concatenated_dataset = Dataset.from_dict({
    'input_ids': concatenated_data['input_ids'],
    'attention_mask': concatenated_data['attention_mask'],
})

mapped_save_path = "/mnt/yiran/cache/Long-Data-Collections-mistral-64k"
sizes = np.array(sizes)
torch.save(sizes, Path(mapped_save_path) / 'sizes.np')
print(concatenated_dataset)

concatenated_dataset.save_to_disk(mapped_save_path)

print("Time", time.time()-s_t)