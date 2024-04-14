import datasets
from transformers import AutoTokenizer
# model_path = "/mnt/yiran/llama-2-7b-hf"
model_path = "/mnt/yiran/Llama2-7b-tokenize"

tokenizer = AutoTokenizer.from_pretrained(model_path)

path_splice = "/mnt/yiran/cache/RedPajama-512k-tokenized"
path_tokenize = "/mnt/yiran/cache/RedPajama-512k-splice"

data_splice = datasets.load_from_disk(path_splice)
data_tokenize = datasets.load_from_disk(path_tokenize)

print("##Comprare first")
# print("data_splice[0]['input_ids'], data_splice[0]['input_ids'][", data_splice[0]['input_ids'])
# print("data_tokenize[0]['input_ids'], data_tokenize[0]['input_ids'][", data_tokenize[0]['input_ids'])
print("##Comprare End")
eg__splice = data_splice[0]['input_ids']
eg__tokenize = data_tokenize[0]['input_ids']

print("Raw data----------------")
ii=0
for i in range(0, len(data_splice[0]['input_ids'])):
    ii = i
    if eg__splice[i] != eg__tokenize[i]:
        print("!!Not the same, id=", i )
        break
print("end ii", ii)

#  6268
print("eg__splice[6268-10: 6268+10]", eg__splice[6268-10: 6268+10])
print("tokenizer.decode(eg__splice[6268-10: 6268+10])", tokenizer.decode(eg__splice[6268-10: 6268+10]) )
print("eg__tokenize[6268-10: 6268+10]", eg__tokenize[6268-10: 6268+10])
print("tokenizer.decode(eg__tokenize[6268-10: 6268+10])", tokenizer.decode(eg__tokenize[6268-10: 6268+10]) )
# Problem
# </s></s><s>
# </s><s>



print("RM eos data----------------")

eg__splice = [x for i, x in enumerate(eg__splice) if x!= 2 or (i == 0 or eg__splice[i-1]!= 2)]
print("len(eg__splice)", len(eg__splice))


ii=0
for i in range(0, len(eg__splice)):
    ii = i
    if eg__splice[i] != eg__tokenize[i]:
        print("!!Not the same, id=", i )
        break
print("end ii", ii)



idx_list = [i for i in range(10)] + [ 300, 500, 1000,]
for idx in idx_list:
    print("\n###############idx", idx)
    splice = data_splice[idx]['input_ids']
    tokenize = data_tokenize[idx]['input_ids']
    print("len(splice)", len(splice))
    print("len(tokenize)", len(tokenize))
    
    print("splice[:5], splice[-5:]", splice[:5], splice[-5:])
    print("tokenizer.decode(splice[:5]), tokenizer.decode(splice[-5:])", tokenizer.decode(splice[:5]), "--", tokenizer.decode(splice[-5:]) )
    
    print("tokenize[:5], tokenize[-5:]", tokenize[:5], tokenize[-5:])
    print("tokenizer.decode(tokenize[:5]), tokenizer.decode(tokenize[-5:])", tokenizer.decode(tokenize[:5]), "--", tokenizer.decode(tokenize[-5:]) )