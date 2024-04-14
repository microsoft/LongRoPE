
from datasets import load_dataset, load_from_disk, Dataset
import transformers
import random
from tqdm import tqdm
random.seed(42)


path = "/mnt/yiran/cache/RedPajama-Data-1T-Sample"
model_path = "/mnt/yiran/Llama-2-7b-hf"
num_proc = 16

short_path = "/mnt/yiran/RedPajama-Data-1T-Sample-less-4k-tokenized"
long_path = "/mnt/yiran/RedPajama-Data-1T-Sample-longer-4k-tokenized"


SEQ_LEN = 4096
LONGROPE_POSE_SEQ = 128*1024
random_times = 6


short_dataset = load_from_disk(short_path)
long_dataset = load_from_disk(long_path)
print(short_dataset, long_dataset)

# short_dataset = short_dataset.select(range(40))
# long_dataset = long_dataset.select(range(40))


# # short data concat:

# tmp_sample = []
# short_dict =  {'input_ids': [], "position_ids": []}
# for data in tqdm(short_dataset):
#     sample = data['input_ids']
#     print("len(tmp_sample)", len(tmp_sample))
#     if len(tmp_sample) < SEQ_LEN:
#         tmp_sample += sample
#     else:
#         short_dict['input_ids'].append(tmp_sample[:SEQ_LEN])
#         short_dict['position_ids'].append([i for i in range(SEQ_LEN)])
#         curr_len = len(tmp_sample)
        
#         tmp_sample = tmp_sample[SEQ_LEN:]
#         print("len(tmp_sample) < SEQ_LEN", len(tmp_sample))

# short_dataset_cat = Dataset.from_dict(short_dict)
# print(short_dataset_cat)
# print(short_dataset_cat[0]['input_ids'][:5], short_dataset_cat[0]['input_ids'][-5:])
# print(short_dataset_cat[0]['position_ids'][:5], short_dataset_cat[0]['position_ids'][-5:])

# short_dataset_cat.save_to_disk("/mnt/yiran/RedPajama-Data-1T-Sample-less-4k-tokenized-cat")


# # long data random:
# tmp_sample = []
# long_dict = {'input_ids': [], "position_ids": []}
# # print(long_dataset[0]['input_ids'])

# for data in tqdm(long_dataset):
#     sample = data['input_ids']
#     assert len(sample) >= SEQ_LEN, f" {len(sample)} >= {SEQ_LEN}"
#     for p in range(random_times):
#         len_input = len(sample)
#         len_chunk = min (SEQ_LEN, len_input)
        
#         lt1 = 0
#         rt1 = random.randint(1, (len_chunk+1)//2)
#         rt2 = random.randint(lt1+len_chunk, len_input)
#         lt2 = rt2 - (len_chunk - (rt1-lt1))
        
#         chunked_ids = sample[lt1:rt1] + sample[lt2:rt2]
#         long_dict['input_ids'].append(chunked_ids)
          
#         pos_ids = [i for i in range(len_chunk)]
#         len_pos_ids = len(pos_ids)
        
#         lt = 0
#         rt = random.randint(lt, LONGROPE_POSE_SEQ-len_pos_ids)
#         # pos_ids[:rt1-lt1] += lt
#         for p in range(rt1-lt1, len_pos_ids):
#             pos_ids[p] += rt
            
#         long_dict['position_ids'].append(pos_ids)
        
# long_dataset_cat = Dataset.from_dict(long_dict)
# print(long_dataset_cat)
# print(long_dataset_cat[0]['input_ids'][:5], long_dataset_cat[0]['input_ids'][-5:])
# print(long_dataset_cat[0]['position_ids'][:5], long_dataset_cat[0]['position_ids'][-5:])
# long_dataset_cat.save_to_disk("/mnt/yiran/RedPajama-Data-1T-Sample-longer-4k-tokenized-random-x6")



# long data no overlap:
chunked_ids_list = []
lr_index = []
long_dict_no_overlap = {'input_ids': [], "position_ids": []}
# print(long_dataset[0]['input_ids'])

for data in tqdm(long_dataset):
    sample = data['input_ids']
    assert len(sample) >= SEQ_LEN, f" {len(sample)} >= {SEQ_LEN}"
    # for p in range(random_times):
    len_input = len(sample)
    len_chunk = min (SEQ_LEN, len_input)
    
    lt1 = 0
    rt1 = random.randint(1, (len_chunk+1)//2)
    rt2 = random.randint(lt1+len_chunk, len_input)
    lt2 = rt2 - (len_chunk - (rt1-lt1))
    
    # generate index
    while rt2 < len_input and (lt1 < rt1 < lt2 < rt2):  
        # print("lt1, rt1, lt2, rt2", lt1, rt1, lt2, rt2)
        chunked_ids = sample[lt1:rt1] + sample[lt2:rt2] 
        assert len(chunked_ids) == len_chunk, f"len(chunked_ids){len(chunked_ids)} != len_chunk {len_chunk}"
        
        chunked_ids_list.append(chunked_ids) 
        lr_index.append([lt1, rt1, lt2, rt2])
        lt1 = rt1 + 1
        if lt1 > lt2:
            break
        rt1 = random.randint(lt1, min(lt2, lt1+(len_chunk+1)//2))  
        lt2 = rt2 + 1
        rt2 = lt2 + (len_chunk - (rt1-lt1))  
        
    # # generate position_ids
    for p in range(len(chunked_ids_list)):
        chunked_ids = chunked_ids_list[p]
        lt1, rt1, lt2, rt2 = lr_index[p]
        
        assert len(chunked_ids) > 1, f"len(chunked_ids)={len(chunked_ids)}"
        pos_ids = [i for i in range(len(chunked_ids))]
        len_pos_ids = len(pos_ids)
        lt = 0
        rt = random.randint(lt, LONGROPE_POSE_SEQ-len_pos_ids)
        
        # pos_ids[:rt1-lt1] += lt
        # pos_ids[rt1-lt1:] += rt
        for p in range(rt1-lt1, len_pos_ids):
            pos_ids[p] += rt

        assert pos_ids[-1] < LONGROPE_POSE_SEQ, f"pos_ids[-1] {pos_ids[-1]} must < {LONGROPE_POSE_SEQ}"
        
        long_dict_no_overlap['input_ids'].append(chunked_ids)
        long_dict_no_overlap['position_ids'].append(pos_ids)
        
        
long_dataset_no_flap = Dataset.from_dict(long_dict_no_overlap)
print(long_dataset_no_flap)
print(long_dataset_no_flap[0]['input_ids'][:5], long_dataset_no_flap[0]['input_ids'][-5:])
print(long_dataset_no_flap[0]['position_ids'][:5], long_dataset_no_flap[0]['position_ids'][-5:])
long_dataset_no_flap.save_to_disk("/mnt/yiran/RedPajama-Data-1T-Sample-longer-4k-tokenized-no-overflap")

