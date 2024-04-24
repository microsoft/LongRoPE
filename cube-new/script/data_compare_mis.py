import datasets
from transformers import AutoTokenizer
# model_path = "/mnt/yiran/llama-2-7b-hf"
model_path = "/mnt/yiran/Mistral-7B-v0.1-16k"

tokenizer = AutoTokenizer.from_pretrained(model_path)

path_512k = "/mnt/yiran/cache/Long-Data-Collections-mistral-512k"


data_512k = datasets.load_from_disk(path_512k)

print(data_512k)
for idx in [0, 1000, 4000, 4500 ]:
    print("############## idx", idx)
    eg__512k = data_512k[idx]['input_ids']
    for i in range(len(eg__512k)-1):
        if eg__512k[i] == 2 :
            print(f"eg__512k[{i-1, i+3}], {eg__512k[i-1: i+3]}")

# print("no 2,2")
# print("eg__512k[6268-10: 6268+10]", eg__512k[6268-10: 6268+10])
# print("tokenizer.decode(eg__512k[6268-10: 6268+10])", tokenizer.decode(eg__512k[6268-10: 6268+10]) )

# print("eg__256k[6268-10: 6268+10]", eg__256k[6268-10: 6268+10])
# print("tokenizer.decode(eg__256k[6268-10: 6268+10])", tokenizer.decode(eg__256k[6268-10: 6268+10]) )

# print("eg__splice[6268-10: 6268+10]", eg__splice[6268-10: 6268+10])
# print("tokenizer.decode(eg__splice[6268-10: 6268+10])", tokenizer.decode(eg__splice[6268-10: 6268+10]) )

# print("eg__tokenize[6268-10: 6268+10]", eg__tokenize[6268-10: 6268+10])
# print("tokenizer.decode(eg__tokenize[6268-10: 6268+10])", tokenizer.decode(eg__tokenize[6268-10: 6268+10]) )