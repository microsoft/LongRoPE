import datasets
from tqdm import tqdm
# path = "/mnt/yiran/cache/RedPajama-512k-tokenized"
path = "/mnt/yiran/cache/RedPajama-128k"

data = datasets.load_from_disk(path)

for i in tqdm(range(data.num_rows)):
    sample = data[i]
    
    if len(sample['input_ids']) != 128*1024:
        print(f"i={i}, len(sample['input_ids'])= {len(sample['input_ids'])}")
        
    # if len(sample['attention_mask'])