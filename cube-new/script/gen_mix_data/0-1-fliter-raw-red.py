from datasets import load_dataset, Dataset
from tqdm import tqdm
import os, json
os.environ['HF_HOME'] = "/mnt/yiran/cache"
# ds = load_dataset("togethercomputer/RedPajama-Data-V2", name="sample")

def gopher_rules_pass(sample) -> bool:
    """ function returns True if the sample complies with Gopher rules """
    signals = json.loads(sample["quality_signals"])

    # rule 1: number of words between 50 and 10'000
    # word_count = signals["rps_doc_word_count"][0][2]
    # if word_count < 50 or word_count > 100_000:
    #     return False

    word_count = signals["rps_doc_word_count"][0][2]
    if word_count <= 131_072:
        return False
    return True


ds_iterator = load_dataset(
    "togethercomputer/RedPajama-Data-V2",
    snapshots=["2023-14"],
    languages=["en"],
    name="default",
    streaming=True,
    trust_remote_code=True,
)

dataset_text = []
dataset_doc_id = []
cnt = 0
num_long = 0
for sample in tqdm(ds_iterator["train"]):
    if cnt % 1000==0: print(cnt)
    cnt += 1
    
    if not gopher_rules_pass(sample):
        continue
    # print(sample.keys())
    print("len(sample)", len(sample['raw_content']))
    num_long += 1
    print("num_long", num_long)
    # filter_sample = {'text': sample['raw_content'], 'doc_id': sample['doc_id']}
    dataset_text.append(sample['raw_content'])
    dataset_doc_id.append(sample['doc_id'])
    
filtered_dataset = Dataset.from_dict({
    'text': dataset_text,
    'doc_id': dataset_doc_id
})

filtered_dataset.save_to_disk("/mnt/yiran/cache/RedPajama-Data-V2-2023-14-longer-128k")
print(filtered_dataset)



