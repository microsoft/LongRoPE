import datasets
from datasets import load_dataset, Dataset, load_from_disk
from tqdm import tqdm
from transformers import AutoTokenizer
from multiprocessing import Pool, cpu_count
import numpy as np
from pathlib import Path
import torch
# import time
# import argparse
import random

random.seed(42)

# s_t = time.time()
# data = datasets.load_from_disk("/mnt/yiran/cache/RedPajama-128k")
# data = datasets.load_from_disk("/mnt/yiran/cache/Long-Data-Collections-mistral-16k")  
# mapped_save_path = "/mnt/yiran/cache/Long-Data-Collections-mistral-64k"
# scale = 4
SEQ_LEN = 4096
LONGROPE_POSE_SEQ = 131072
    
def get_tokenizer(tokenizer_name_or_path,
                  model_max_length=None,
                  default_bos_token="<s>",
                  default_eos_token="</s>",
                  default_pad_token="[PAD]",
                  default_unk_token="<unk>"):

    tokenizer = AutoTokenizer.from_pretrained(
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
  

def doc12x(idx, sample, tokenizer):
    inputs = sample["text"]
    raw_model_inputs = tokenizer(
        inputs, 
        padding=False, 
        truncation=True, 
        max_length=LONGROPE_POSE_SEQ,
        return_tensors='pt'
        ) 
    # ['input_ids', 'attention_mask']
    local_data_dict = {'input_ids': [], 'attention_mask': [], 'position_ids': []}  
    ids = raw_model_inputs['input_ids'].tolist() # [1 * seq_len]
    ids = ids[0] # [ seq_len ]
    # get short data:
    if len(ids) < SEQ_LEN:
        if idx < 640000//2 :
            # pad
            padding_len = SEQ_LEN - len(ids)
            input_ids = ids + [tokenizer.pad_token_id] * padding_len
            attention_mask = [1] * len(ids) + [0] * padding_len
            position_ids = [i for i in range(len(ids))] + [len(ids)-1] * padding_len
            
            # append
            assert len(input_ids) == len(attention_mask) and len(input_ids) == len(position_ids), f"{len(input_ids), len(attention_mask), len(position_ids)}"
            local_data_dict['input_ids'].append(input_ids)  
            local_data_dict['attention_mask'].append(attention_mask)  
            local_data_dict['position_ids'].append(position_ids)  
        else:
            pass
    else:   
        # get longdata x6:
        chunked_ids_list = []
        lr_index = []
        len_chunk = SEQ_LEN
        len_input = len(ids) # [seq_len]
        
        # generate idx
        lt1 = 0
        rt1 = random.randint(1, (len_chunk+1)//2)
        rt2 = random.randint(lt1+len_chunk, len_input)
        lt2 = rt2 - (len_chunk - (rt1-lt1))
        while rt2 < len_input:  
            # print("lt1, rt1, lt2, rt2", lt1, rt1, lt2, rt2)
            chunked_ids = ids[lt1:rt1] + ids[lt2:rt2] 
            assert len(chunked_ids) == len_chunk, f"len(chunked_ids){len(chunked_ids)} != len_chunk {len_chunk}"
            
            chunked_ids_list.append(chunked_ids) 
            lr_index.append([lt1, rt1, lt2, rt2])
            lt1 = rt1 + 1
            rt1 = random.randint(lt1, min(lt2, lt1+(len_chunk+1)//2))  

            lt2 = rt2 + 1
            rt2 = lt2 + (len_chunk - (rt1-lt1))
        
        # generate position_ids
        for p in range(len(chunked_ids_list)):
            chunked_ids = chunked_ids_list[p]
            lt1, rt1, lt2, rt2 = lr_index[p]
            
            assert len(chunked_ids) > 1, f"len(chunked_ids)={len(chunked_ids)}"
            pos_ids = torch.arange(0, len(chunked_ids))
            len_pos_ids = len(pos_ids)
            lt = 0
            rt = random.randint(lt, LONGROPE_POSE_SEQ-len_pos_ids)
            
            pos_ids[:rt1-lt1] += lt
            pos_ids[rt1-lt1:] += rt

            assert pos_ids[-1] < LONGROPE_POSE_SEQ, f"pos_ids[-1] {pos_ids[-1]} must < {LONGROPE_POSE_SEQ}"
            
            pos_ids = pos_ids.tolist()
            
            # append
            input_ids = chunked_ids
            attention_mask = [1] * SEQ_LEN
            position_ids = pos_ids
            
            assert len(input_ids) == len(attention_mask) and len(input_ids) == len(position_ids)
            local_data_dict['input_ids'].append(input_ids)
            local_data_dict['attention_mask'].append(attention_mask)
            local_data_dict['position_ids'].append(position_ids)
        
    return local_data_dict

def process_chunk(chunk):  
    # 对于数据块中的每一行，调用 process_data 函数  
    # 并将结果收集在列表中返回  
    result_chunk = []  
    for args in chunk:  
        result = process_data(args)  
        result_chunk.append(result)  
    return result_chunk  

def process_data(args):  
    index, data, tokenizer = args
    # 假设 doc12x 是一个函数，它接受索引、数据和tokenizer，然后返回一个更新的数据字典  
    # 需要确保 doc12x 函数是可以序列化的，否则它不能在多进程中使用  
    return doc12x(index, data, tokenizer)  
  
def merge_dicts(dicts):  
    # 初始化合并后的字典  
    merged_dict = {'input_ids': [], 'attention_mask': [], 'position_ids': []}  
    # 遍历列表中的每个字典  
    for d in dicts:  
        # 将每个局部字典的内容追加到合并后的字典中  
        for key in merged_dict:  
            merged_dict[key].extend(d[key])  
    return merged_dict  
  
if __name__ == '__main__':  # 这是多进程编程的必备条件  
    # 准备数据  
    data_name_or_path = "/mnt/yiran/cache/RedPajama-Data-1T-Sample"
    tokenizer_id = "/mnt/yiran/Llama-2-7b-hf"
    mapped_save_path = "/mnt/yiran/cache/RedPajama-4k-pose-128k-same-doc2-fliter-12xdoc"
    try:
        dataset = load_dataset(path=data_name_or_path, cache_dir="/mnt/yiran/cache")
    except:
        dataset = load_from_disk(data_name_or_path)
    train_dataset = dataset['train']
    
    # select 40 rows to test
    # train_dataset = train_dataset.select(range(400))
    
    max_seq_len = SEQ_LEN
    tokenizer = get_tokenizer(tokenizer_id, max_seq_len)
        
    data_to_process = [(i, train_dataset[i], tokenizer) for i in range(train_dataset.num_rows)]  
    
    num_full_batches = len(data_to_process)
    
    num_proc = int(cpu_count()*0.8)
    print("num_proc", num_proc)
    
    chunk_size = len(data_to_process) // num_proc  
    chunks = [data_to_process[i:i + chunk_size] for i in range(0, len(data_to_process), chunk_size)]  
     # 为了确保数据完整性，处理可能遗留的余数部分  
    if len(data_to_process) % num_proc != 0:  
        # 将剩余的数据添加到最后一个块中  
        chunks[-1].extend(data_to_process[len(chunks) * chunk_size:])  
      
    # 创建进程池  
    # with Pool(num_proc) as pool:  
    #     # 处理数据  
    #     # results = pool.starmap(process_data, data_to_process)  
    #     results = list(tqdm(pool.imap(process_data, data_to_process), desc='Processing data', total=num_full_batches))
    
    with Pool(num_proc) as pool:  
        # 使用 tqdm 包裹 imap 函数来创建一个有进度条的迭代器  
        results = list(tqdm(pool.imap(process_chunk, chunks), desc='Processing data', total=len(chunks)))  
            
    print("merge dict")
    # 合并结果  
    # data_dict = merge_dicts(results)  
    # 现在 data_dict 已经被所有进程的结果更新 
    merged_results = [item for sublist in results for item in sublist]  
    data_dict = merge_dicts(merged_results)  
    
    
    # convert to tp
    input_ids_tp= [torch.tensor(x, dtype=torch.long) for x in data_dict['input_ids']]
    attention_mask_tp= [torch.tensor(x, dtype=torch.int8) for x in data_dict['attention_mask']]
    position_ids_tp= [torch.tensor(x, dtype=torch.long) for x in data_dict['position_ids']]
    
    # convert dict to Dataset
    train_dataset = Dataset.from_dict({
        "input_ids": input_ids_tp,
        "attention_mask": attention_mask_tp,
        "position_ids": position_ids_tp,
    }).with_format("torch")
    # 
    train_dataset.save_to_disk(mapped_save_path)

    sizes = np.full((train_dataset.num_rows,), SEQ_LEN)
    torch.save(sizes, Path(mapped_save_path) / 'sizes.np')

