import os  
import time  
import subprocess  
  
from pathlib import Path 
import re
import argparse

# 设置checkpoint目录和合并脚本的路径  
# proj_dir = "/mnt/yiran/cube-new/LongRoPE/cube-new/cube-test"
# storage = "https://fastnn.blob.core.windows.net/teamdrive/ExtendSeqLen/ft_out_model/cube-la2-4k-128k-same-doc/test"
# key = "?sv=2023-01-03&st=2024-04-22T12%3A15%3A28Z&se=2024-04-25T12%3A15%3A00Z&sr=c&sp=racwdl&sig=lQc6Vz%2BpSby%2BbHMe%2B64MLO1hXEKiroRXAio7DhvIvSE%3D"  

parser = argparse.ArgumentParser()
parser.add_argument('--proj-dir', type=str, required=True)
parser.add_argument('--sh-path', type=str, required=True)
parser.add_argument('--storage', type=str, required=True)
parser.add_argument('--key', type=str, required=True)
args = parser.parse_args()

proj_dir = args.proj_dir
storage = args.storage
key = args.key
sh_path = args.sh_path

print(f"proj_dir={proj_dir}, \nsh_path={sh_path} \nkey={key}, \nstorage={storage}, ")
checkpoint_dir = proj_dir + "/checkpoint"  
 

checkpoint_list = []
checkpoint_step_pattern = r"checkpoint_(.+)-shard0.pt"

# checkpoint_list = ["1_100", "1_200", "1_300", "1_400", "1_500", "1_600", "1_700", "1_800", "1_900", "1_1000"]

# 设置监视的间隔时间（单位秒）  
interval = int(3600*0.5)  # 每小时检查一次  
pt_split = 4

# checkpoint文件模式  
checkpoint_pattern = "checkpoint_{}-shard{}.pt"  

def merge_checkpoints(checkpoint_step, checkpoint_dir, sh_path, storage, key):  
    ck = checkpoint_step
    print(f"Processing checkpoint: {ck}")  
    
    # 创建目录
    ck_dir = Path(checkpoint_dir) / f"ck-{ck}"
    if (ck_dir / "pytorch_model.bin").exists(): 
        return False
    
    # 合并checkpoint片段  
    merge_command = f"bash {sh_path} mergeckpt {checkpoint_dir}/checkpoint_{ck}-shard0.pt"  
    print(merge_command)
    subprocess.run(merge_command, shell=True, check=True)  

    # 提取为Hugging Face模型格式  
    extract_command = f"bash {sh_path} extract_hf {checkpoint_dir}/checkpoint_{ck}-full.pt"  
    print(extract_command)
    subprocess.run(extract_command, shell=True, check=True)  

    # 移动模型文件  
      
    ck_dir.mkdir(exist_ok=True)  
    pytorch_model_file = Path(checkpoint_dir) / "pytorch_model.bin"  
    print(f"mv to {ck_dir / pytorch_model_file.name}")
    pytorch_model_file.rename(ck_dir / pytorch_model_file.name)  

    # 使用azcopy上传到云存储  
    azcopy_command = f"azcopy cp '{ck_dir.as_posix()}/' '{storage}/{key}' --recursive=true"  
    print(azcopy_command)
    subprocess.run(azcopy_command, shell=True, check=True)  

    # 检查模型文件是否存在，如果存在则删除相关checkpoint文件  
    if (ck_dir / "pytorch_model.bin").exists():  
        for shard_id in range(pt_split):  # 假设有8个shard  
            shard_file = Path(checkpoint_dir) / f"checkpoint_{ck}-shard{shard_id}.pt"  
            shard_file.unlink(missing_ok=True)  
        full_pt_file = Path(checkpoint_dir) / f"checkpoint_{ck}-full.pt"  
        full_pt_file.unlink(missing_ok=True)  
        
    return True
  


# 开始监视循环  
while True:  
    
    for filename in os.listdir(checkpoint_dir):
        print(filename)
        match  = re.match(checkpoint_step_pattern, filename)
        if match:
            ck_step = match.group(1)
            print(ck_step)
            checkpoint_list.append(ck_step)
    checkpoint_list.sort()
    print(checkpoint_list)
    # exit(0)
    # 检查每个checkpoint文件  
    for checkpoint_step in checkpoint_list:  # 假设步长为100，总步数为1000 
        correct_pt = 0
        for id in range(pt_split): # checkpoint_1_100-shard0.pt - 7.pt
            checkpoint_file = os.path.join(checkpoint_dir, checkpoint_pattern.format(checkpoint_step, id))  
            
            if not os.path.exists(checkpoint_file):  
                print(f"Checkpoint {checkpoint_file} not found. Waiting for next check.")  
                break  
            else:
                correct_pt += 1
        if correct_pt == pt_split:
            # Merge
            # 如果所有文件都存在，则执行合并脚本  
            print(f"All {checkpoint_step} ck found. Running merge script.")  

            # merge 
            flag = merge_checkpoints(checkpoint_step, checkpoint_dir, sh_path, storage, key)  
            
            if flag == False:
                print(f"{checkpoint_step} exits")
            else:
                print(f"{checkpoint_step} OK")
                
            if checkpoint_step == checkpoint_list[-1]:
                print("Merge script executed. Exiting monitor script.")  
                break  # 退出监视循环  
  
    # 等待指定的间隔时间  
    time.sleep(interval)  
