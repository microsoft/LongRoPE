from transformers import AutoTokenizer
from datasets import load_from_disk, concatenate_datasets, Dataset
import re, random
import numpy as np  

PG19_DATPATH = "pg19"  # replace with your path to the PG19 dataset
SEQLEN = 128 * 1024
t1 = "A special magic number is hidden within the following text. Make sure to memorize it. I will quiz you about the number afterwards.\n"
t2 = "One of the special magic numbers for wandering-age is: {}.\n"
t3 = "The special magic number for wandering-age mentioned in the provided text is: {}"

def find_all_indices(string, substring):
    return [match.start() for match in re.finditer(re.escape(substring), string)]

def find_closest_index(nums, target):
    nums_array = np.array(nums)
    index = (np.abs(nums_array - target)).argmin()
    return index

tokenizer = AutoTokenizer.from_pretrained("Meta-Llama-3-8B", trust_remote_code=True)

def tokenized(sample, key='text'):
    input_ids = tokenizer.encode(sample[key])
    return {'input_ids': input_ids}

dataset = load_from_disk(PG19_DATPATH)
dataset = dataset.filter(lambda a: len(tokenized(a)['input_ids']) >= SEQLEN, num_proc=64)
break_token_ids = [tokenizer.encode('\n', add_special_tokens=False)[0], tokenizer.encode(',\n', add_special_tokens=False)[0], tokenizer.encode('.\n', add_special_tokens=False)[0]]

input_ids_list = []
cus_label_list = []

for i in range(10):
    text = dataset[i]['text']
    tokenized_ids = tokenizer.encode(text, add_special_tokens=False)
    break_list = [_ for _, num in enumerate(tokenized_ids) if num in break_token_ids]
    for j in range(10):
        break_idx = int(SEQLEN * (i + j * 10) / 99)
        break_idx_idx = find_closest_index(break_list, break_idx)
        break_idx = break_list[break_idx_idx] if break_list[break_idx_idx] < SEQLEN else break_list[break_idx_idx-1]
        front, back = tokenized_ids[:break_idx+1], tokenized_ids[break_idx+1:SEQLEN]
        # print(f"break_idx: {break_idx}, front len: {len(front)}, back len: {len(back)}")
        needle = str(random.randint(1000000, 9999999))
        needle_ids = tokenizer.encode(str(needle), add_special_tokens=False)
        input_ids = [tokenizer.bos_token_id] + tokenizer.encode(t1, add_special_tokens=False) + front + tokenizer.encode(t2.format(needle), add_special_tokens=False) + back + tokenizer.encode(t3.format(needle), add_special_tokens=False)
        cus_label = [-100] * (len(input_ids) - len(needle_ids)) + needle_ids
        # print(f"end ids: {input_ids[-5:]}\nlabel ids: {cus_label[-5:]}\ntotal len: {len(input_ids)}")
        assert len(input_ids) == len(cus_label)
        input_ids_list.append(input_ids)
        cus_label_list.append(cus_label)

dataset = Dataset.from_dict({'input_ids': input_ids_list, 'cus_label': cus_label_list})
dataset.save_to_disk("pg19-train-128k-search-llama3-tokenized-hf")
