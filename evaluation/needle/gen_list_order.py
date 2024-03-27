import itertools  
import random  
import json  
  
from prompt import file_list_pt
# 假设这是你的原始列表，此处仅作演示用途  
# file_list_pt = 
print("file_list_pt", len(file_list_pt))
# 创建一个字典来存储排列  
file_list_dict = {}  
  
# 随机生成100个唯一的排列  
unique_permutations = set()  
while len(unique_permutations) < 100:  
    # 随机打乱列表  
    shuffled_list = file_list_pt[:]  
    random.shuffle(shuffled_list)  
    # 将列表转换为元组（因为列表是不可哈希的，不能作为集合的元素）  
    shuffled_tuple = tuple(shuffled_list)  
    # 添加到集合中保持唯一性  
    unique_permutations.add(shuffled_tuple)  
  
# 将原始未打乱的列表作为第一个条目添加到字典中  
file_list_dict[0] = file_list_pt  
  
# 将排列添加到字典中  
for index, permutation in enumerate(unique_permutations, start=1):  
    file_list_dict[index] = list(permutation)  
  
# 将字典保存到JSON文件中，包含缩进  
with open('file_list_dict.json', 'w') as json_file:  
    json.dump(file_list_dict, json_file, indent=4)  
  
print("已将排列保存到 file_list_dict.json，包含缩进")  
