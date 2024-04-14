import datasets
from transformers import AutoTokenizer
# model_path = "/mnt/yiran/llama-2-7b-hf"
model_path = "/mnt/yiran/Llama2-7b-tokenize"

tokenizer = AutoTokenizer.from_pretrained(model_path)

path_128k = "/mnt/yiran/cache/RedPajama-128k"
path_256k = "/mnt/yiran/cache/RedPajama-256k"
path_splice = "/mnt/yiran/cache/RedPajama-512k-tokenized"
path_tokenize = "/mnt/yiran/cache/RedPajama-512k-splice"

data_128k = datasets.load_from_disk(path_128k)
data_256k = datasets.load_from_disk(path_256k)
data_splice = datasets.load_from_disk(path_splice)
data_tokenize = datasets.load_from_disk(path_tokenize)

eg__128k = data_128k[0]['input_ids']
eg__256k = data_256k[0]['input_ids']
eg__splice = data_splice[0]['input_ids']
eg__tokenize = data_tokenize[0]['input_ids']

# index = [0:5] + []


# print("eg__128k[6268-10: 6268+10]", eg__128k[6268-10: 6268+10])
# print("tokenizer.decode(eg__128k[6268-10: 6268+10])", tokenizer.decode(eg__128k[6268-10: 6268+10]) )

# print("eg__256k[6268-10: 6268+10]", eg__256k[6268-10: 6268+10])
# print("tokenizer.decode(eg__256k[6268-10: 6268+10])", tokenizer.decode(eg__256k[6268-10: 6268+10]) )

# print("eg__splice[6268-10: 6268+10]", eg__splice[6268-10: 6268+10])
# print("tokenizer.decode(eg__splice[6268-10: 6268+10])", tokenizer.decode(eg__splice[6268-10: 6268+10]) )

# print("eg__tokenize[6268-10: 6268+10]", eg__tokenize[6268-10: 6268+10])
# print("tokenizer.decode(eg__tokenize[6268-10: 6268+10])", tokenizer.decode(eg__tokenize[6268-10: 6268+10]) )