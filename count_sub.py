import matplotlib.pyplot as plt  
  
# 数据集字典  
datasets = {  
    'NI_decontaminated_materialized': {'0-4096': 39323, '4096-8192': 331695, '8192-16384': 205930, '16384-32768': 725, '32768-65536': 310, '65536-131072': 60, '131072-262144': 0, '262144+': 0},  
    'pile_sub': {'0-4096': 1808968, '4096-8192': 68044, '8192-16384': 46322, '16384-32768': 12116, '32768-65536': 2047, '65536-131072': 1544, '131072-262144': 1150, '262144+': 267},  
    'P3_decontaminated_materialized': {'0-4096': 0, '4096-8192': 2969, '8192-16384': 811017, '16384-32768': 29, '32768-65536': 0, '65536-131072': 0, '131072-262144': 0, '262144+': 0},  
    'rp_sub': {'0-4096': 880966, '4096-8192': 30274, '8192-16384': 13168, '16384-32768': 4485, '32768-65536': 1079, '65536-131072': 373, '131072-262144': 155, '262144+': 14}  
}  
  
# 创建图表和子图  
fig, axs = plt.subplots(2, 2, figsize=(21, 14), dpi=400)  
  
# 子图标题  
titles = [  
    'NI Decontaminated Materialized',  
    'Pile Subset',  
    'P3 Decontaminated Materialized',  
    'RP Subset'  
]  
  
# 遍历数据集和子图  
for ax, (name, len_dict) in zip(axs.flatten(), datasets.items()):  
    total = sum(len_dict.values())  # 计算每个数据集的总长度  
    keys = len_dict.keys()  
    values = len_dict.values()  
  
    # 创建条形图  
    ax.bar(keys, values, color='skyblue')  
    ax.set_title(name)  
    ax.set_xlabel('Token length ranges')  
    ax.set_ylabel('Number of samples')  
  
    # 在条形图上显示数值  
    for i, (key, value) in enumerate(len_dict.items()):  
        value_p = (value / total) * 100  # 计算百分比  
        percentage = "{:d}, {:.2f}%".format(value, value_p)  # 格式化为保留两位小数的百分比字符串  
        ax.text(i, value, percentage, ha='center', va='bottom', fontsize=12)  
  
# 调整子图布局  
plt.tight_layout()  
  
# 保存图像  
plt.savefig("dataset_distributions.png")  
  
# 显示图表  
plt.show()  
