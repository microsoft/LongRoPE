import pandas as pd  
import json  
import os  
  
# 文件名模板和变量  
# ck_steps = ["1_1000", "1_900", "1_800", "1_700", "1_600", "1_500", "1_400", "1_300", "1_200", "1_100"] 
ck_steps = ["6_1000", "5_900", "5_800", "4_700", "4_600", "3_500", "3_400", "2_300", "2_200", "1_100"]
job_names = ["ARC", "HELLASWAG", "MMLU", "TRUTHFULQA"]  
  
# 初始化一个空的字典，用来存储结果数据  
results_dict = {job_name: [] for job_name in job_names}  
  
# 遍历所有可能的文件名组合  
for ck_step in ck_steps:  
    for job_name in job_names:  
        # /mnt/yiran/LongRoPE/script/hf_benchmark/ft_out_model/longrope-256k-sft/from-step-600/ck-3_400/-ARC-longrope-bs2_la2_256k_step1000_ck-3_400.json
        file_template = "-{job_name}-longrope-bs2_la2_256k_step1000_ck-{ck_step}.json"  
        file_name = file_template.format(ck_step=ck_step, job_name=job_name)  
        file_path = f'/mnt/yiran/LongRoPE/script/hf_benchmark/ft_out_model/longrope-256k-sft/from-step-1000/ck-{ck_step}/{file_name}'  
          
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
results_df = pd.DataFrame(results_dict, index=ck_steps)  
  
print(results_df)  
