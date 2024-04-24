import argparse
import copy
from functools import partial
from itertools import chain

from pathlib import Path
from tqdm import tqdm
import torch
import numpy as np
import random
import os



from datasets import load_dataset, load_from_disk
from transformers import DataCollatorForLanguageModeling
from fairseq.data import FairseqDataset

# from .utils import get_tokenizer
from data_helper.utils import get_tokenizer

# from cube_examples.finetune_hf_model.src.data_helper.utils import get_tokenizer

IGNORE_IDX = -100

random.seed(42)
def hf_tokenize_map_fn(dataset, tokenizer, split, feature, seq_len, num_proc=16):
    # split dataset
    dataset = dataset[split] 
    # redpajama only have on split 'train'
    dataset = dataset.remove_columns([x for x in dataset.column_names if x not in [feature]])

    # Add bos eos
    tokenized_dataset = dataset.map(
        lambda example: tokenizer(
            [t + tokenizer.eos_token for t in example[feature]]),
        batched=True,
        num_proc=num_proc,
        remove_columns=[feature],
    )

    # seq_len 128k
    block_size = seq_len 

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {
            k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + block_size]
                for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        return result

    if split == "train":
        tokenized_dataset = tokenized_dataset.map(
            group_texts, batched=True, num_proc=num_proc,
        )

    return tokenized_dataset

class WrappedHFDataset(FairseqDataset):
    def __init__(self, data_name_or_path: str,
                 cache_dir: str,
                 tokenizer_id: str,
                 max_seq_len: int,
                 mapped_save_path: str,
                 if_tokenize: bool = False,
                 shuffle: bool = False,
                 data_split: str = 'train',
                 data_feature: str = 'text',
                 data_num_proc: int = 16,
                 ) -> None:
        
        tokenizer = get_tokenizer(tokenizer_id, max_seq_len)
        
        if if_tokenize:
            try:
                dataset = load_dataset(path=data_name_or_path, cache_dir=cache_dir)
            except:
                dataset = load_from_disk(data_name_or_path)
                
            self.dataset = None
            
            if 'DATA_FUNC' in os.environ:
                data_func = os.environ['DATA_FUNC']
                print("data_func", data_func)
            else:
                raise ValueError(f"Use DATA_FUNC in [Redpajama_tokenize, Long_Data_from_yarn, Redpajama_Pose, Extend_Input_ids_Seq, Redpajama_Same_Doc] ")
            # Redpajama
            if data_func == 'Redpajama_tokenize':
                print("Data - Redpajama")
                data_num_proc = 4
                self.dataset = hf_tokenize_map_fn(dataset=dataset, tokenizer=tokenizer, split='train', feature='text', seq_len=max_seq_len, num_proc=data_num_proc)
                
            # Long Red
            if data_func == 'Long_Data_from_yarn':
                print("Data - Long Red")
                train_dataset = load_dataset(path=data_name_or_path, cache_dir=cache_dir)
                print(train_dataset)
                print(len(train_dataset['train'][0]['input_ids']))
                from datasets import DatasetDict
                if isinstance(train_dataset, DatasetDict):
                    train_dataset = train_dataset["train"]
                if "input_ids" not in train_dataset.column_names:
                    raise RuntimeError("Dataset must include an `input_ids` feature")
                if "labels" not in train_dataset.column_names:
                    def add_labels(sample):
                        sample["labels"] = copy.deepcopy(sample["input_ids"])
                        return sample
                    train_dataset = train_dataset.map(
                        add_labels, desc="Adding labels", num_proc=args.num_proc)
                if "attention_mask" not in train_dataset.column_names:
                    def add_attention_mask(sample):
                        sample["attention_mask"] = torch.ones(
                            len(sample["input_ids"]), dtype=torch.int8)
                        return sample
                    train_dataset = train_dataset.map(
                        add_attention_mask, desc="Adding attention mask", num_proc=args.num_proc)
            
                self.dataset = train_dataset
            
            # Redpajama Pose
            if data_func == 'Redpajama_Pose':
                print("Data - Redpajama Pose")
                train_dataset = dataset
                if "input_ids" not in train_dataset.column_names:
                    raise RuntimeError("Dataset must include an `input_ids` feature")
                    
                print("$input_ids-old", train_dataset[0]['input_ids'][:5], train_dataset[0]['input_ids'][-5:])
                
                if "position_ids" not in train_dataset.column_names:
                    def add_position_ids(sample):
                        # PoSE
                        model_max_position_embeddings = int(os.environ['SEQ_LEN']) 
                        # 128*1024
                        scaled_max_position_embeddings = int(os.environ['LONGROPE_POSE_SEQ'])
                        # 512*1024
                        
                        raw_model_inputs = sample["input_ids"]
                        # input_ids = []
                        # position_ids = []
                        
                        ids = raw_model_inputs
                        len_chunk = min(len(ids), model_max_position_embeddings)
                        len_input = len(ids)
                        lt1 = 0
                        rt1 = random.randint(1, (len_chunk+1)//2)
                        rt2 = random.randint(lt1+len_chunk, len_input)
                        lt2 = rt2 - (len_chunk - (rt1-lt1))
                        chunked_ids = ids[lt1:rt1] + ids[lt2:rt2]
                        # input_ids.append(chunked_ids)
                        
                        sample["input_ids"] = torch.tensor(chunked_ids)
                        
                        pos_ids = torch.arange(len(chunked_ids), dtype=torch.long)
                        len_pos_ids = len(pos_ids)
                        # lt = random.randint(0, scaled_max_position_embeddings-len_pos_ids)
                        lt = 0 # this revision makes the coverage possiblity more uniform for large relative positions
                        rt = random.randint(lt, scaled_max_position_embeddings-len_pos_ids)

                        pos_ids[:rt1-lt1] += lt
                        pos_ids[rt1-lt1:] += rt
                        # position_ids.append(pos_ids)
                        assert pos_ids[-1] < scaled_max_position_embeddings, f"pos_ids[-1] {pos_ids[-1]} must < {scaled_max_position_embeddings}"
                        sample["position_ids"] = torch.tensor(pos_ids)

                        return sample
                    
                    print("Use PoSE")
                    print("num_rows", train_dataset.num_rows)
                    model_max_position_embeddings = int(os.environ['SEQ_LEN']) 
                    scaled_max_position_embeddings = int(os.environ['LONGROPE_POSE_SEQ'])
                    print("model_max_position_embeddings", model_max_position_embeddings)
                    print("scaled_max_position_embeddings", scaled_max_position_embeddings)
                    
                    train_dataset = train_dataset.map(
                        add_position_ids, desc="Adding position_ids", num_proc=data_num_proc)
                    
                    print("$position_ids", train_dataset[5]['position_ids'][:5], train_dataset[0]['position_ids'][-5:], len(train_dataset[5]['position_ids']))
                    print("$input_ids-new", train_dataset[5]['input_ids'][:5], train_dataset[0]['input_ids'][-5:], len(train_dataset[5]['input_ids']))
                    self.dataset = train_dataset
                
            # Extend seq
            if data_func == 'Extend_Input_Ids_Seq':
                print("Data - Redpajama Pose")
                train_dataset = dataset
                print("Colums: ", train_dataset.column_names)
                origin_data_len = len(train_dataset['train'][0]['input_ids'])
                print("origin_data_len", origin_data_len)
                print("max_seq_len", max_seq_len)
                
                def extend_len(sample, max_seq_len, column_names):
                    # copy 128k-> max_seq_len=512k
                    assert max_seq_len % (128*1024) == 0, "max_seq_len must for 128k*n"
                    for col in column_names:
                        sample[col] = torch.cat(sample[col] * 4, dim=0)
                    
                    return sample
                train_dataset = train_dataset.map(
                    extend_len, 
                    desc="extend_len", 
                    num_proc=args.num_proc,
                    fn_kwargs={"max_seq_len": max_seq_len,
                               "column_names": train_dataset.column_names})
                
            # Redpajama -> same doc
            if data_func == 'Redpajama_Same_Doc':
                print("Data - Redpajama same doc")
                print(dataset)
                dataset = dataset['train']

                train_dataset = dataset
                  
                # print("$input_ids-old", input_ids[0]['input_ids'][:5], input_ids[0]['input_ids'][-5:])
                
                if "position_ids" not in train_dataset.column_names:
                    def add_position_ids(sample, tokenizer):
                        # PoSE
                        model_max_position_embeddings = int(os.environ['SEQ_LEN']) 
                        # 128*1024
                        scaled_max_position_embeddings = int(os.environ['LONGROPE_POSE_SEQ'])
                        # 512*1024
                        inputs = sample["text"]

                        raw_model_inputs = tokenizer(
                            inputs, 
                            padding=False, 
                            truncation=True, max_length=model_max_position_embeddings*5,
                            return_tensors='pt'
                            )
                        
                        ids = raw_model_inputs['input_ids'] # [1 * seq_len]
                        
                        len_chunk = min(ids.shape[1], model_max_position_embeddings)
                        len_input = ids.shape[1]
                        lt1 = 0
                        rt1 = random.randint(1, (len_chunk+1)//2)
                        rt2 = random.randint(lt1+len_chunk, len_input)
                        lt2 = rt2 - (len_chunk - (rt1-lt1))
                        
                        # chunked_ids = ids[lt1:rt1] + ids[lt2:rt2]
                        chunked_ids = torch.cat((ids[0, lt1:rt1], ids[0, lt2:rt2]))
                        # input_ids.append(chunked_ids)

                        # sample["input_ids"] = torch.tensor(chunked_ids)
                        sample["input_ids"] = chunked_ids.squeeze(0).clone().detach()
                        
                        pos_ids = torch.arange(len(chunked_ids), dtype=torch.long)
                        len_pos_ids = len(pos_ids)
                        # lt = random.randint(0, scaled_max_position_embeddings-len_pos_ids)
                        lt = 0 # this revision makes the coverage possiblity more uniform for large relative positions
                        rt = random.randint(lt, scaled_max_position_embeddings-len_pos_ids)

                        pos_ids[:rt1-lt1] += lt
                        pos_ids[rt1-lt1:] += rt
                        # position_ids.append(pos_ids)
                        assert pos_ids[-1] < scaled_max_position_embeddings, f"pos_ids[-1] {pos_ids[-1]} must < {scaled_max_position_embeddings}"
                        sample["position_ids"] = pos_ids.clone().detach()

                        # exit(0)
                        return sample
                    
                    print("Use PoSE")
                    print("num_rows", train_dataset.num_rows)
                    model_max_position_embeddings = int(os.environ['SEQ_LEN']) 
                    scaled_max_position_embeddings = int(os.environ['LONGROPE_POSE_SEQ'])
                    print("model_max_position_embeddings", model_max_position_embeddings)
                    print("scaled_max_position_embeddings", scaled_max_position_embeddings)
                    
                    train_dataset = train_dataset.map(
                        add_position_ids, 
                        desc="Adding position_ids", 
                        # num_proc=data_num_proc,
                        num_proc=4,
                        remove_columns="text",
                        fn_kwargs={"tokenizer": tokenizer}
                        )
                    
                    
                    print("$position_ids", train_dataset[5]['position_ids'][:5], train_dataset[5]['position_ids'][-5:], len(train_dataset[5]['position_ids']))
                    print("$input_ids-new", train_dataset[5]['input_ids'][:5], train_dataset[5]['input_ids'][-5:], len(train_dataset[5]['input_ids']))
                    
                    self.dataset = train_dataset
                
            if data_func == 'Redpajama_Same_Doc_Fliter':
                train_dataset = dataset
                def filter_func(x):
                    return len(x['input_ids']) >= int(os.environ['SEQ_LEN']) 
                def add_attention_mask(sample):
                    sample["attention_mask"] = torch.ones(
                        len(sample["input_ids"]), dtype=torch.int8)
                    return sample
                print("fliter short")
                train_dataset = train_dataset.filter(filter_func, desc="fliter short", num_proc=data_num_proc)
                train_dataset = train_dataset.map(add_attention_mask, desc="Adding attention mask", num_proc=data_num_proc)
                print("$train_dataset", train_dataset)
                
                self.dataset = train_dataset
                
            if data_func == 'Redpajama_Same_Doc_Pad':
                train_dataset = dataset
                print(train_dataset)
                SEQ_LEN = int(os.environ['SEQ_LEN']) 
                def pad_input_ids(x):
                    # print(type(x))
                    input_ids_len = len(x['input_ids'])
                    x["attention_mask"] = [1] * input_ids_len
                    # torch.ones(len(x["input_ids"]), dtype=torch.int64)
                    if input_ids_len < SEQ_LEN:
                        padding_len = SEQ_LEN - input_ids_len
                        x['input_ids'] += [tokenizer.pad_token_id] * padding_len  # pad with zeros
                        x['attention_mask'] += [0] * padding_len  # pad attention mask as well
                        pos_last = x['position_ids'][-1]
                        x['position_ids'] += [pos_last] * padding_len
                        # print(x['input_ids'][:5], x['input_ids'][-5:],)
                        # print(x['attention_mask'][:5], x['attention_mask'][-5:],)
                        # exit(0)
                    return x
                
                train_dataset = train_dataset.map(pad_input_ids, num_proc=data_num_proc)
                
                print(train_dataset[5]['input_ids'][:5], train_dataset[5]['input_ids'][-5:],)
                print(train_dataset[5]['attention_mask'][:5], train_dataset[5]['attention_mask'][-5:],)
                print(train_dataset[5]['position_ids'][:5], train_dataset[5]['position_ids'][-5:],)
                self.dataset = train_dataset
                
            if self.dataset is None:
                raise ValueError("Should `self.dataset = train_dataset`")
            
            self.dataset.save_to_disk(mapped_save_path)
            
            sizes = []
            for data in tqdm(self.dataset):
                sizes.append(len(data['input_ids']))
            self.sizes = np.array(sizes)
            torch.save(self.sizes, Path(mapped_save_path) / 'sizes.np')
        else:
            self.dataset = load_from_disk(mapped_save_path)
            self.sizes = torch.load(Path(mapped_save_path) / 'sizes.np')
            print("self.dataset[0]['input_ids'].shape", len(self.dataset[0]['input_ids']))
        self.shuffle = shuffle
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        """id is needed in fairseq"""
        data = self.dataset.__getitem__(int(i))
        data['id'] = i
        return data

    def size(self, index):
        return self.sizes[index]

    def num_tokens(self, index):
        return self.sizes[index]

    def ordered_indices(self):
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]
        order.append(self.sizes)
        return np.lexsort(order)

    def collater(self, samples):
        if len(samples) == 0:
            return {}

        mini_batch = self.data_collator(samples)
        _mini_batch = {}
        _mini_batch['src_tokens'] = mini_batch.pop('input_ids')
        _mini_batch['labels'] = torch.nn.functional.pad(mini_batch['labels'][..., 1:], (0, 1), 'constant', IGNORE_IDX).contiguous()
        if os.environ['POSITION_POSE'] == "true":
            # print("POSITION_POSE")
            _mini_batch['position_ids'] = mini_batch.pop('position_ids')
        else:
            # print("Not POSITION_POSE")
            pass
            
        # print("$ _mini_batch['position_ids']",  _mini_batch('position_ids'))
        # _mini_batch['attention_mask'] = mini_batch['attention_mask']
        # _mini_batch['id'] = mini_batch['id']

        return {
            "id": torch.LongTensor([s['id'] for s in samples]),
            "nsentences": len(samples),
            "ntokens": sum(self.sizes[s['id']] for s in samples),
            "net_input": _mini_batch,
            "target": _mini_batch.pop('labels'),
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name_or_path')
    parser.add_argument('--cache_dir')
    parser.add_argument('--tokenizer_id')
    parser.add_argument('--max_seq_len')
    parser.add_argument('--mapped_save_path')
    parser.add_argument('--if_tokenize')
    parser.add_argument('--data_split')
    parser.add_argument('--data_feature')
    parser.add_argument('--data_num_proc')
    
    args = parser.parse_args()

    if Path(args.data_name_or_path).is_dir():
        if any(Path(args.data_name_or_path).rglob("*.py")):
            args.data_name_or_path = str(next(Path(args.data_name_or_path).rglob("*.py")))

    print("if_tokenize", args.if_tokenize)
    WrappedHFDataset(data_name_or_path=args.data_name_or_path,
                     cache_dir=args.cache_dir,
                     tokenizer_id=args.tokenizer_id,
                     max_seq_len=int(args.max_seq_len),
                     mapped_save_path=args.mapped_save_path,
                     if_tokenize=args.if_tokenize,
                     shuffle=True,
                     data_split=args.data_split,
                     #  data_feature=args.data_feature,
                     #  data_num_proc=args.data_num_proc
                    )
