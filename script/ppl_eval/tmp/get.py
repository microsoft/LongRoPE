import numpy as np    
import pandas as pd    
    
config_list = ["longrope_128k", "longrope_256k", "longrope_128k_mistral", "longrope_256k_mistral"]    
seq_len_list = [4096, 8192]    
tmps_list = [0.8, 0.85, 0.9, 0.95, 1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.1, 1.11, 1.12, 1.13, 1.14, 1.15, 1.2, 1.25, 1.3]    
# 初始化DataFrame    
dataframes = {config: pd.DataFrame(index=seq_len_list, columns=tmps_list) for config in config_list}    
    
# 填充DataFrame    
for config in config_list:    
    for seq_len in seq_len_list:    
        for tmps in tmps_list:    
            f_name = f"script/ppl_eval/tmp/proofpile_{config}_{seq_len}_{tmps}.csv"    
                
            # 读取CSV文件    
            try:    
                data = pd.read_csv(f_name, header=None, usecols=[1])    
                value = float(data.iloc[-1, 0]) # 假设值总是在第二列的最后一行    
                value = round(value, 5)  
                # 将值填入对应的位置    
                dataframes[config].loc[seq_len, tmps] = value    
            except FileNotFoundError:    
                print(f"File {f_name} not found.")    
                dataframes[config].loc[seq_len, tmps] = np.nan  # 如果文件不存在，填入NaN    
  
# 保存每个DataFrame到CSV文件    
for config, df in dataframes.items():    
    # 为每个配置创建CSV文件名    
    csv_file_name = f"{config}_results.csv"    
    # 保存DataFrame到CSV    
    df.to_csv(csv_file_name)    
    print(f"Config {config} saved to {csv_file_name}")  
