import torch
import numpy as np

# load rope_para:
# ft: 4k 8k 256k 512k 1024k 
finetuned = True
method = "longrope"
# max_tokens = 16*1024
# model_type = "mistral"
max_position_embeddings = 256*1024
change_keys = []
# mis_256_path = "/mnt/yiran/2048k-mistral-256k/s-PI/evolution/test/result_alpha/mistral_262144_dim_mono_ppl9.csv"
# mis_256 = torch.from_numpy(np.loadtxt(open(mis_256_path, "rb"), delimiter=",", skiprows=0))

# rope_rescale = torch.load("./evaluation/rope_rescale-new.pt")
# rope_rescale['ft_mis_256k'] = mis_256

print(rope_rescale.keys())
# exit(0)
for max_tokens in [ 512*1024, 1024*1024, 2048*1024]:
    model_type = "mistral"
    print("max_tokens", max_tokens)
    if finetuned and method == "longrope":
        print("finetuned", finetuned, "use rope_scale.pt")
        if max_tokens != None:
            seq_len = (max_tokens + 1023) // 1024
            seq_range = [0, 4, 8, 16, 32, 128, 256, 512, 1024, 2048, 10000]
            for i in range(len(seq_range)-1):
                if seq_range[i] <= seq_len <= seq_range[i+1]:   
                    seq_len = seq_range[i+1]
                    break
            if model_type == "mistral": 
                model_type = "mis"
            else:
                raise ValueError("model_type is not llama")  
            ft_model_len = (max_position_embeddings + 1023) // 1024

            flag_twice = False
            ft_model_key = None
            
            if seq_len == ft_model_len:
                para_key = f"ft_{model_type}_{ft_model_len}k"
            elif seq_len > ft_model_len:
                para_key = f"{seq_len}k_{model_type}_{ft_model_len}k"
                flag_twice = True
                ft_model_key = f"ft_{model_type}_{ft_model_len}k"
            else:
                para_key = f"{seq_len}k_{model_type}_{ft_model_len}k"
            
            # 128k la2 256k
            if para_key == '128k_la2_256k':
                para_key = 'ft_la2_256k'
            print("para_key", para_key)
            
            print("flag_twice", flag_twice)
            if flag_twice:
                lambda_1 = rope_rescale[para_key] * rope_rescale[ft_model_key]
                rope_rescale[para_key] = lambda_1
                change_keys.append(para_key)
            else: 
                lambda_1 = rope_rescale[para_key]
            
    # rope_rescale = torch.load("./evaluation/rope_rescale.pt")
    # print(rope_rescale.keys())
    print(lambda_1)
    
print("end----------------")
change_keys.append("8k_la2_128k")
change_keys.append('ft_la2_256k')

for key in change_keys:
    print(key)
    print(rope_rescale[key])
print(rope_rescale["1024k_la2_256k"].dtype)

## save
file_path = "evaluation/rope_rescale-new.pt"
torch.save(rope_rescale, file_path)

# reload
rope_rescale_new = torch.load(file_path)
print(rope_rescale_new["1024k_la2_256k"].dtype)

for key in change_keys:
    print(key)
    print(rope_rescale_new[key])