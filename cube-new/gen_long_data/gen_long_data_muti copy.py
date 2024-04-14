import datasets
from datasets import load_dataset, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from multiprocessing import Pool, cpu_count
import numpy as np
from pathlib import Path
import torch
import time


def process_batch(i):
        batch_input_ids = sum([data[i+j]['input_ids'] for j in range(scale)], [])
        batch_attention_mask = sum([data[i+j]['attention_mask'] for j in range(scale)], [])
        if model_type == 'mistral':
            batch_attention_labels = sum([data[i+j]['labels'] for j in range(scale)], [])
        return batch_input_ids, batch_attention_mask, batch_attention_labels


def process_data(source_data, mapped_save_path, scale, model_type):
    s_t = time.time()
    data = datasets.load_from_disk(source_data)
    print(data)

    num_full_batches = len(data) // scale
    print(data)
    print("origin data rows", len(data))
    print("new data rows", num_full_batches)

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

    
    num_processes = cpu_count() - 1
    
    with Pool(processes=num_processes) as pool:  
        results = list(tqdm(pool.imap(lambda i: process_batch(data, scale, model_type, i), range(0, num_full_batches * scale, scale)), desc='Concatenating data', total=num_full_batches))  
    
    for result in results:
        batch_input_ids, batch_attention_mask, batch_attention_labels = result
        concatenated_data['input_ids'].append(batch_input_ids)
        concatenated_data['attention_mask'].append(batch_attention_mask)
        concatenated_data['labels'].append(batch_attention_labels)
        sizes.append(len(batch_input_ids))

    print("$len", len(concatenated_data['input_ids'][0]))

    concatenated_dataset = Dataset.from_dict({
        'input_ids': concatenated_data['input_ids'],
        'attention_mask': concatenated_data['attention_mask'],
    })

    sizes = np.array(sizes)
    torch.save(sizes, Path(mapped_save_path) / 'sizes.np')
    print(concatenated_dataset)

    concatenated_dataset.save_to_disk(mapped_save_path)

    print("Time", time.time()-s_t)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--source_data", type=str, required=True)
    parser.add_argument("--mapped_save_path", type=str, required=True)
    parser.add_argument("--scale", type=int, required=True)

    args = parser.parse_args()

    if 'RedPajama' in args.source_data:
        model_type = "llama2"
    elif 'Long-Data-Collections' in args.source_data:
        model_type = "mistral"
    else:
        raise ValueError("model_type not supports")
    
    process_data(args.source_data, args.mapped_save_path, args.scale, model_type)
