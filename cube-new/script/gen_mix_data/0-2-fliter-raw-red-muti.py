from datasets import load_dataset  
from torch.utils.data import DataLoader  
from tqdm import tqdm  
import os, json  
from multiprocessing import cpu_count  

# 设置环境变量  
os.environ['HF_HOME'] = "/mnt/yiran/cache"  
  
def gopher_rules_pass(sample) -> bool:  
    signals = json.loads(sample["quality_signals"])  
    word_count = signals["rps_doc_word_count"][0][2]  
    return word_count >= 131_072  

def collate_fn(batch):  
    filtered_batch = []  
    for sample in batch:  
        if gopher_rules_pass(sample):  
            filtered_batch.append({  
                'text': sample['raw_content'],  
                'doc_id': sample['doc_id']  
            })  
    return filtered_batch  
  
# 初始化数据集迭代器  
ds_iterator = load_dataset(  
    "togethercomputer/RedPajama-Data-V2",  
    snapshots=["2023-14"],  
    languages=["en"],  
    name="default",  
    streaming=True,  
    trust_remote_code=True  
)["train"]  
  
# 创建PyTorch DataLoader  
data_loader = DataLoader(  
    ds_iterator,  
    batch_size=1,  # 根据需求调整批量大小  
    collate_fn=collate_fn,  
    # num_workers=cpu_count()-4  # 使用所有可用的CPU核心
    num_workers=8
)  
  
# 收集过滤后的样本  
dataset_text = []  
dataset_doc_id = []  
  
num_long = 0
# 使用tqdm来显示进度  
for batch_samples in tqdm(data_loader, desc="Processing samples"):  
    for sample in batch_samples: 
        num_long += 1
        print("num_long", num_long)
        print("len(sample['text'])", len(sample['text']))
        dataset_text.append(sample['text'])  
        dataset_doc_id.append(sample['doc_id'])  
  
# 创建一个新的数据集  
filtered_dataset = {  
    'text': dataset_text,  
    'doc_id': dataset_doc_id  
}  
  
# 在这里你可以保存filtered_dataset到磁盘或者进行其他处理  

# 保存到磁盘  
filtered_dataset.save_to_disk("/mnt/yiran/cache/RedPajama-Data-V2-2023-14-longer-128k-parallel")  
print(filtered_dataset)  