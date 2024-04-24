import pandas as pd  
import json  
import os  
  
# 文件名模板和变量  
# /mnt/yiran/LongRoPE-main/LongRoPE/script/hf_benchmark/ft_out_model/cube-mis-256k-bf16-step-500/ck-1_350/-{job_name}-longrope-bs2_mis_256k_bf16-step-500_step{ck_step}.json

# ck_steps = ["1_1000", "1_900", "1_800", "1_700", "1_600", "1_500", "1_400", "1_300", "1_200", "1_100"]
# ck_steps = ["1_500", "1_450", "1_400", "1_350", "1_300", "1_250", "1_200", "1_150", "1_100", "1_50"]
tmps_steps = ["1.0"]
job_names = ["ARC", "HELLASWAG", "MMLU", "TRUTHFULQA"]  
  
# 初始化一个空的字典，用来存储结果数据  
results_dict = {job_name: [] for job_name in job_names}  
  
# 遍历所有可能的文件名组合  
for tmps in tmps_steps:  
    for job_name in job_names:  
        # /mnt/yiran/LongRoPE-main/LongRoPE/script/hf_benchmark/tmp-search/Llama-2-7b-hf/-MMLU-pi-bs2_la2_4k_tmps1.0.json
        file_template = "-{job_name}-pi-bs2_la2_4k_tmps{tmps}.json"  
        file_name = file_template.format(tmps=tmps, job_name=job_name)  
        file_path = f'script/hf_benchmark/tmp-search/Llama-2-7b-hf/{file_name}'  
        
        # 初始设置为NaN，以防文件不存在或无法读取数据  
        result = pd.NA  
          
        # 检查文件是否存在  
        if os.path.exists(file_path):  
            # 打开并读取JSON文件  
            with open(file_path, 'r', encoding='utf-8') as file:  
                data = json.load(file)  
                  
                # 根据job_name选择要提取的结果  
                if job_name == "TRUTHFULQA":  
                    result = data['results']['truthfulqa_mc']['mc2']  
                elif job_name == "MMLU":  
                    values = [item["acc_norm"] for item in data['results'].values()]  
                    result = sum(values) / len(values)  
                elif job_name == "ARC":  
                    result = data['results']['arc_challenge']['acc_norm']  
                else:  
                    result = data['results'][job_name.lower()]['acc_norm']  
        else:
            print(f"file_path is not exit {file_path}")
        results_dict[job_name].append(result)  
  
# 使用字典创建一个DataFrame，其中字典的键是列名，值是列数据  
results_df = pd.DataFrame(results_dict, index=tmps_steps)  
  
print(results_df)  
