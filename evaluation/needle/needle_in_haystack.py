"""
This script is adapted from 
https://github.com/gkamradt/LLMTest_NeedleInAHaystack

# GPT-4
(
python -u needle_in_haystack.py --s_len 0 --e_len 128000\
    --model_provider OpenAI\
    --model_name gpt-4-1106-preview
    --api_key $OPENAI_API_KEY
) 2>&1  | tee logs/eval_gpt_4_128k.log

# LLaMA 2 32K. Remember to download the model first
(
python -u needle_in_haystack.py --s_len 0 --e_len 128000\
    --model_provider LLaMA\
    --model_path ../../../Llama-2-7B-32K-Instruct
) 2>&1  | tee logs/eval_llama2_32k_instruct.log

# LongChat. Remember to download the model first
(
python -u needle_in_haystack.py --s_len 0 --e_len 128000\
    --model_provider LLaMA\
    --model_path /ML-A800/models/longchat-7b-v1.5-32k
) 2>&1  | tee logs/eval_longchat.log

# Our llama-2-7b-80k, requires 4*80G A100
# require you to download the model first
(
python -u needle_in_haystack.py --s_len 0 --e_len 128000\
    --model_provider LLaMA\
    --model_path ../../../llama-2-7b-80k
) 2>&1  | tee logs/eval_llama-2-7b-80k.log
"""


import os 
import glob
import json
import tensor_parallel as tp
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import cube

try:
    import tiktoken
    from anthropic import Anthropic
    from openai import OpenAI
except:
    print("!! Requirement not support using API of openai & anthropic.\n if you want, please `pip install openai tiktoken anthropic`")
#from dotenv import load_dotenv
import numpy as np
import argparse
from rouge_score import rouge_scorer

import sys
current_path = os.getcwd()
sys.path.append(current_path)
print(current_path)


scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

import torch.distributed as dist  
  
def print_single(*args):  
    if dist.get_rank() == 0:  
        for args_output in args:  
            print(args_output)  
  

# print_single("Hello", "Distributed", "World")  


from datetime import datetime, timezone
import time
import torch

class LLMNeedleHaystackTester:
    """
    This class is used to test the LLM Needle Haystack.
    """
    def __init__(self,
                 needle="\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n",
                 haystack_dir="PaulGrahamEssays",
                 retrieval_question="What is the best thing to do in San Francisco?",
                 results_version = 1,
                 context_lengths_min = 1000,
                 context_lengths_max = 128000,
                 context_lengths_num_intervals = 40,
                 context_lengths = None,
                 document_depth_percent_min = 0,
                 document_depth_percent_max = 100,
                 document_depth_percent_intervals = 10,
                 document_depth_percents = None,
                 document_depth_percent_interval_type = "linear",
                 model_provider = "OpenAI",
                 openai_api_key=None,
                 anthropic_api_key = None,
                 model_name='',
                 model_name_suffix=None,
                 num_concurrent_requests = 1,
                 save_results = True,
                 save_contexts = True,
                 result_path = 'result',
                 use_cache = False,
                 final_context_length_buffer = 200,
                 seconds_to_sleep_between_completions = None,
                 print_ongoing_status = True,
                 args_rope = None,
                 prompt_template="base",
                 ):
        """        
        :param needle: The needle to be found in the haystack. Default is None.
        :param haystack_dir: The directory of text files to use as background context (or a haystack) in which the needle is to be found. Default is Paul Graham Essays.
        :param retrieval_question: The question which with to prompt the model to do the retrieval.
        :param results_version: In case you would like to try the same combination of model, context length, and depth % multiple times, change the results version other than 1
        :param num_concurrent_requests: Due to volume, this object is set up to run concurrent requests, default = 1. Be careful of rate limits.
        :param save_results: Whether or not you would like to save your contexts to file. Warning: These will get long! Default = True
        :param save_contexts: Whether or not you would like to save your contexts to file. Warning: These will get long! Default is True.
        :param save_contexts: choose a path to save result.
        :param final_context_length_buffer: The amount of cushion you'd like to leave off the input context to allow for the output context. Default 200 tokens
        :param context_lengths_min: The minimum length of the context. Default is 1000.
        :param context_lengths_max: The maximum length of the context. Default is 200000.
        :param context_lengths_num_intervals: The number of intervals for the context length. Default is 35.
        :param context_lengths: The lengths of the context. Default is None.
        :param document_depth_percent_min: The minimum depth percent of the document. Default is 0.
        :param document_depth_percent_max: The maximum depth percent of the document. Default is 100.
        :param document_depth_percent_intervals: The number of intervals for the document depth percent. Default is 35.
        :param document_depth_percents: The depth percentages of the document. Default is None.
        :param document_depth_percent_interval_type: The type of interval for the document depth percent. Must be either 'linear' or 'sigmoid'. Default is 'linear'.
        :param model_provider: The provider of the model. Must be either 'OpenAI' or 'Anthropic'. Default is 'OpenAI'.
        :param openai_api_key: The API key for OpenAI. Default is None.
        :param anthropic_api_key: The API key for Anthropic. Default is None.
        :param model_name: The name of the model. Default is 'gpt-4-1106-preview'.
        :param seconds_to_sleep_between_completions: The number of seconds to sleep between completions. Default is None.
        :param print_ongoing_status: Whether or not to print the ongoing status. Default is True.
        """
        if not needle or not haystack_dir or not retrieval_question:
            raise ValueError("Needle, haystack, and retrieval_question must be provided.")
        
        self.needle = needle
        self.haystack_dir = haystack_dir
        self.retrieval_question = retrieval_question
        self.results_version = results_version
        self.num_concurrent_requests = num_concurrent_requests
        self.save_results = save_results
        self.final_context_length_buffer = final_context_length_buffer
        self.save_contexts = save_contexts
        self.result_path = result_path
        self.use_cache = use_cache
        self.seconds_to_sleep_between_completions = seconds_to_sleep_between_completions
        self.print_ongoing_status = print_ongoing_status
        self.model_provider = model_provider
        self.testing_results = []
        
        self.args_rope = args_rope
        self.prompt_template = prompt_template
        if("/" in model_name):
            self.model_version = model_name.split("/")[-1]
        else: self.model_version = model_name
        if(model_name_suffix is not None): self.model_version += "_" + model_name_suffix

        if context_lengths is None:
            if context_lengths_min is None or context_lengths_max is None or context_lengths_num_intervals is None:
                raise ValueError("Either context_lengths_min, context_lengths_max, context_lengths_intervals need to be filled out OR the context_lengths_list needs to be supplied.")
            else:
                self.context_lengths = np.round(np.linspace(context_lengths_min, context_lengths_max, num=context_lengths_num_intervals, endpoint=True)).astype(int)
        else:
            self.context_lengths = context_lengths

        if document_depth_percents is None:
            if document_depth_percent_min is None or document_depth_percent_max is None or document_depth_percent_intervals is None:
                raise ValueError("Either document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals need to be filled out OR the document_depth_percents needs to be supplied.")
            else:
                if document_depth_percent_interval_type == 'linear':
                    self.document_depth_percents = np.round(np.linspace(document_depth_percent_min, document_depth_percent_max, num=document_depth_percent_intervals, endpoint=True)).astype(int)
                elif document_depth_percent_interval_type == 'sigmoid':
                    self.document_depth_percents = [self.logistic(x) for x in np.linspace(document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals)]
        else:
            self.document_depth_percents = document_depth_percents

        if document_depth_percent_interval_type not in [None, "linear", "sigmoid"]:
            raise ValueError("document_depth_percent_interval_type must be either None, 'linear' or 'sigmoid'. If you'd like your own distribution give a list of ints in via document_depth_percent_intervals")
        
        self.model_name = model_name

        if(self.model_provider not in ["OpenAI", "Anthropic"]):
            self.enc = AutoTokenizer.from_pretrained(model_name, use_fast = True )
            print("loading from %s" % model_name)

            if args.use_cube:
                if self.args_rope.method == "longrope":
                    
                    config = AutoConfig.from_pretrained(model_name)
                    
                    if args.cube_trace:
                        if config.model_type == "mistral":
                            print(model_name)
                            from evaluation.model_loader_mistral_cube import load_model_and_apply_patches_mistral
                            self.model_to_test = load_model_and_apply_patches_mistral(model_name, config, self.args_rope)
                        elif config.model_type == "llama":
                            print(model_name)
                            from evaluation.model_loader_llama_cube import load_model_and_apply_patches
                            self.model_to_test = load_model_and_apply_patches(model_name, config, self.args_rope)
                        else:
                            raise ValueError("Model type did not support!")
                    else:
                        if config.model_type == "mistral":
                            print(model_name)
                            from evaluation.model_loader_mistral_cube import update_config
                        elif config.model_type == "llama":
                            from evaluation.model_loader_llama_cube import update_config
                        else:
                            raise ValueError("Model type did not support!")
                        self.model_to_test = None
                    # new_config
                    self.config = update_config(config, self.args_rope)
                    from evaluation.cube_api import compile_model
                    self.model_to_test, self.infer_fn = compile_model(self.model_to_test, self.args_rope, self.config)
            else:
                self.infer_fn = None
                # self.model_to_test = None
                if self.args_rope.method != "longrope":
                    # default setting
                    self.model_to_test = AutoModelForCausalLM.from_pretrained(model_name,
                        use_flash_attention_2="flash_attention_2", 
                        torch_dtype=torch.bfloat16,
                        device_map="auto",
                        ).eval()

                    self.model_to_test = tp.tensor_parallel(self.model_to_test, sharded=True)

            # compute_perplexity
            # longlora-100k
 
            # "--tokenized ${/mnt/yiran/teamdrive3/ExtendSeqLen}/proofpile-test-tokenized --dataset_min_tokens 131072 --samples 10 --truncate"
            # import datasets
            # input_texts = datasets.load_from_disk("/mnt/yiran/teamdrive3/ExtendSeqLen/proofpile-test-tokenized")
            
            # input_texts = input_texts.filter(
            #     lambda x: x["tokenized_len"] >= 131072)
            
            # input_texts = input_texts[ :10 ]
            # tokenizer=AutoTokenizer.from_pretrained(self.model_name, use_fast = True )
            # from evaluation.perplexity import compute_perplexity
            # max_length = 8192
            # print_single("self.model_to_test", self.model_to_test)
            # ppl = compute_perplexity(
            #     model=self.model_to_test, tokenizer=tokenizer, encodings=input_texts,
            #     add_start_token=tokenizer.bos_token is not None, max_length=max_length,
            #     sliding_window=256, truncate=True,
            #     use_cache=False
            #     )['mean_perplexity']
            # print_single(f"$max_length {max_length} ppl {ppl}")
            # exit(0)
        
        else: 
            self.model_to_test = OpenAI(api_key=openai_api_key)
            if(self.model_provider == "OpenAI"):
                self.enc = tiktoken.encoding_for_model(self.model_name)
            elif(self.model_provider == "Anthropic"):
                self.enc = Anthropic().get_tokenizer()

        self.model_to_test_description = model_name
        
        self.evaluation_model = None
        self.debug='debug'
        model_name = model_name.split('/')[-1]

    def logistic(self, x, L=100, x0=50, k=.1):
        if x == 0:
            return 0
        if x == 100:
            return 100
        return np.round(L / (1 + np.exp(-k * (x - x0))), 3)
    
    def bound_evaluate_and_log(self, *args):
        self.evaluate_and_log(*args)

    def run_test(self, args):

        # Run through each iteration of context_lengths and depths
        tasks = []
        for context_length in self.context_lengths:
            if context_length < args.s_len or context_length > args.e_len: 
                continue
            
            # load model
            # change
            if not args.use_cube:
                if self.args_rope.method == "longrope":
                    # diff seq
                    self.args_rope.max_tokens = context_length
                    # self.args_rope.max_tokens = 128000
                    
                    config = AutoConfig.from_pretrained(model_name)
                    if config.model_type == "mistral":
                        print(model_name)
                        from evaluation.model_loader_mistral import load_model_and_apply_patches_mistral
                        loaded, _ = load_model_and_apply_patches_mistral(model_name, self.args_rope)
                        self.model_to_test = loaded.eval()
                        
                    elif config.model_type == "llama":
                        print(model_name)
                        from evaluation.model_loader_llama import load_model_and_apply_patches
                        loaded, _ = load_model_and_apply_patches(model_name, self.args_rope)
                        self.model_to_test = loaded.eval()
                        
                    else:
                        raise ValueError("Model type did not support!")
                    
                    self.model_to_test = tp.tensor_parallel(self.model_to_test, sharded=True)
            
            for depth_percent in self.document_depth_percents:
                task = self.bound_evaluate_and_log(context_length, depth_percent)

    def generate_prompt(self, context):
        # Generate the prompt for the Anthropic model
        # Replace the following line with the appropriate prompt structure
        if self.prompt_template == "base":
            if(self.model_provider not in ["OpenAI", "Anthropic"]):
                test_format=f"<|im_start|> This is a very long story book: <book> {context} </book>.\n Based on the content of the book, Question: {self.retrieval_question}\nAnswer:"
                return test_format
            else: 
                return [
                    {
                        "role": "system",
                        "content": "You are a helpful AI bot that answers questions for a user. Keep your response short and direct"
                    },
                    {
                        "role": "user",
                        "content": context
                        },
                    {
                        "role": "user",
                        "content": f"{self.retrieval_question} Don't give information outside the document or repeat your findings. The document definitely contains the answer, and I'm 100% sure. So try your best to find it."
                    },
                    {
                        "role": "assistant",
                        "content":"",
                    },
                    
                ]
        elif self.prompt_template == "SIMPLE_TEMPLATE":
            from prompt import SIMPLE_TEMPLATE
            test_format = SIMPLE_TEMPLATE.format(question=self.retrieval_question, context=context)
            return test_format
        elif self.prompt_template == "ANTHROPIC_TEMPLATE_REV1":
            from prompt import ANTHROPIC_TEMPLATE_REV1
            test_format = ANTHROPIC_TEMPLATE_REV1.format(question=self.retrieval_question, context=context)
            return test_format
        elif self.prompt_template == "ANTHROPIC_TEMPLATE_REV2":
            from prompt import ANTHROPIC_TEMPLATE_REV2
            test_format = ANTHROPIC_TEMPLATE_REV2.format(question=self.retrieval_question, context=context)
            return test_format
        elif self.prompt_template == "ANTHROPIC_TEMPLATE_ORIGINAL":
            from prompt import ANTHROPIC_TEMPLATE_ORIGINAL
            test_format = ANTHROPIC_TEMPLATE_ORIGINAL.format(question=self.retrieval_question, context=context)
            return test_format
        elif self.prompt_template == "GEMINI_TEMPLATE":
            from prompt import GEMINI_TEMPLATE
            test_format = GEMINI_TEMPLATE.format(question=self.retrieval_question, context=context)
            return test_format
        elif self.prompt_template == "GEMINI_TEMPLATE2":
            from prompt import GEMINI_TEMPLATE2
            test_format = GEMINI_TEMPLATE2.format(question=self.retrieval_question, context=context)
            return test_format
        elif self.prompt_template == "ANTHROPIC_TEMPLATE_REV1_ED":
            from prompt import ANTHROPIC_TEMPLATE_REV1_ED
            test_format = ANTHROPIC_TEMPLATE_REV1_ED.format(question=self.retrieval_question, context=context)
            return test_format
    
    def evaluate_and_log(self, context_length, depth_percent):
        # Checks to see if you've already checked a length/percent/version.
        # This helps if the program stop running and you want to restart later
        if self.save_results:
            if self.result_exists(context_length, depth_percent):
                print_single("result exists, skipping")
                return
            else:
                print_single("result does not exist, testing")
                
        print_single("begin generate_context")
        # Go generate the required length context and place your needle statement in
        context = self.generate_context(context_length, depth_percent)

        print_single("begin generate_prompt")
        # Prepare your message to send to the model you're going to evaluate
        prompt = self.generate_prompt(context)
        
        print_single("$rm bos")
        prompt = prompt.replace("<s>", "")
        print_single(prompt)
        
        test_start_time = time.time()
        if(self.model_provider in ["OpenAI", "Anthropic"]):
            # import ipdb; ipdb.set_trace()
            response = self.model_to_test.chat.completions.create(
                model=self.model_name,
                messages=prompt,
                max_tokens=300,
                temperature=0
            )
            response = response.choices[0].message.content
        else:
            print_single("begin tokenizer")
            # print_single("prompt[:, -100:]", prompt[-100:])
            prompt = self.enc(prompt, return_tensors="pt")
            print_single("end tokenizer")
            
            input_ids = prompt['input_ids'].to(torch.cuda.current_device())
            if input_ids[0, -1] == self.enc.eos_token_id:
                input_ids = input_ids[:, :-1]
            
            print_single("input_ids[:, -5:]", input_ids[:, -5:])
            
            print_single("begin generate, context_length", context_length)
            
            # test ppl
            with torch.no_grad():
                # output_ids = self.model_to_test.generate(input_ids, max_new_tokens=50, use_cache=self.use_cache)
                if args.use_cube:
                    from evaluation.cube_api import generate
                    response = generate(self.args_rope, self.model_to_test, self.infer_fn, self.config, self.enc, input_ids, max_new_tokens=32)
                else:
                    output_ids = self.model_to_test.generate(input_ids, max_new_tokens=32, use_cache=self.use_cache)
                    response = self.enc.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
            print_single("end generate")
                
        test_end_time = time.time()
        test_elapsed_time = test_end_time - test_start_time
        
        print_single(f"$$self.needle,\"{self.needle}\"\nresponse,\"{response}\"\n")
        score = scorer.score(self.needle, response)['rouge1'].fmeasure*10
        
        results = {
            # 'context' : context, # Uncomment this line if you'd like to save the context the model was asked to retrieve from. Warning: This will become very large.
            'model' : self.model_to_test_description,
            'context_length' : int(context_length),
            'depth_percent' : float(depth_percent),
            'version' : self.results_version,
            'needle' : self.needle,
            'model_response' : response,
            'score' : score,
            'test_duration_seconds' : test_elapsed_time,
            'test_timestamp_utc' : datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S%z')
        }

        self.testing_results.append(results)

        if self.print_ongoing_status:
            print_single (f"-- Test Summary -- ")
            print_single (f"Duration: {test_elapsed_time:.1f} seconds")
            print_single (f"Context: {context_length} tokens")
            print_single (f"Depth: {depth_percent}%")
            print_single (f"Score: {score}")
            print_single (f"Response: {response}\n")

        context_file_location = f'{self.model_version.replace(".", "_")}_len_{context_length}_depth_{int(depth_percent*100)}'

        if self.save_contexts:
            if args.use_cube:
                if torch.distributed.get_rank() == 0:
                    results['file_name'] : context_file_location

                    # Save the context to file for retesting
                    if not os.path.exists('contexts'):
                        os.makedirs('contexts')

                    if not os.path.exists(f'contexts/{self.model_version}'):
                        os.makedirs(f'contexts/{self.model_version}')

                    # with open(f'contexts/{self.model_version}/{context_file_location}_context.txt', 'w') as f:
                    #     f.write(context)
                    with open(f'contexts/{self.model_version}/{context_file_location}_context.txt', 'w', encoding='utf-8') as f:  
                        f.write(context)  
            else:
                results['file_name'] : context_file_location

                # Save the context to file for retesting
                if not os.path.exists('contexts'):
                    os.makedirs('contexts')

                if not os.path.exists(f'contexts/{self.model_version}'):
                    os.makedirs(f'contexts/{self.model_version}')

                # with open(f'contexts/{self.model_version}/{context_file_location}_context.txt', 'w') as f:
                #     f.write(context)
                with open(f'contexts/{self.model_version}/{context_file_location}_context.txt', 'w', encoding='utf-8') as f:  
                    f.write(context)  
            
        if self.save_results:
            # Save the context to file for retesting
            
            if args.use_cube:
                if torch.distributed.get_rank() == 0:
                    if not os.path.exists(self.result_path):
                        os.makedirs(self.result_path)
                    
                    if not os.path.exists(f'{self.result_path}/{self.model_version}'):
                        os.makedirs(f'{self.result_path}/{self.model_version}')

                    # Save the result to file for retesting
                    p = f'{self.result_path}/{self.model_version}/{context_file_location}_results.json'
                    print_single("Writing at %s" % p)
                    with open(p, 'w') as f:
                        json.dump(results, f)
            else:
                if not os.path.exists(self.result_path):
                    os.makedirs(self.result_path)
                
                if not os.path.exists(f'{self.result_path}/{self.model_version}'):
                    os.makedirs(f'{self.result_path}/{self.model_version}')

                # Save the result to file for retesting
                p = f'{self.result_path}/{self.model_version}/{context_file_location}_results.json'
                print_single("Writing at %s" % p)
                with open(p, 'w') as f:
                    json.dump(results, f)

    def result_exists(self, context_length, depth_percent):
        """
        Checks to see if a result has already been evaluated or not
        """

        results_dir = 'results/' + self.model_version
        print_single("Searching existing results at %s" % results_dir)
        if not os.path.exists(results_dir):
            return False
        for filename in os.listdir(results_dir):
            if filename.endswith('.json'):
                with open(os.path.join(results_dir, filename), 'r') as f:
                    result = json.load(f)
                    context_length_met = result['context_length'] == context_length
                    depth_percent_met = result['depth_percent'] == depth_percent
                    version_met = result.get('version', 1) == self.results_version
                    model_met = result['model'] == self.model_name
                    
                    if context_length_met and depth_percent_met and version_met and model_met:
                        return True
        return False

    def generate_context(self, context_length, depth_percent):
        # Load up tiktoken so we navigate tokens more easily

        # Get your Paul Graham files loaded into a string
        context = self.read_context_files()

        # Truncate the Paul Graham essays to the context length you desire
        context = self.encode_and_trim(context, context_length)

        # Insert your random statement according to your depth percent
        context = self.insert_needle(context, depth_percent, context_length)

        return context
    
    def encode_text_to_tokens(self, text):
        if self.model_provider in ["OpenAI", "LLaMA", "Mistral", "GLM"]:
            return self.enc.encode(text)
        elif self.model_provider == "Anthropic":
            # Assuming you have a different encoder for Anthropic
            return self.enc.encode(text).ids
        else:
            raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")
    
    def insert_needle(self, context, depth_percent, context_length):
        tokens_needle = self.encode_text_to_tokens(self.needle)
        tokens_context = self.encode_text_to_tokens(context)

        print_single("$before tokens_needle[:5], tokens_needle[:-5]", tokens_needle[:5], tokens_needle[:-5])
        if tokens_needle[-1] == self.enc.eos_token_id:
            tokens_needle = tokens_needle[:-1]
        print_single("$after tokens_needle[:5], tokens_needle[:-5]", tokens_needle[:5], tokens_needle[:-5])
        
        # Reducing the context length by 150 buffer. This is to account for system message, the user question, and response.
        context_length -= self.final_context_length_buffer

        # If your context + needle are longer than the context length (which it will be), then reduce tokens from the context by the needle length
        if len(tokens_context) + len(tokens_needle) > context_length:
            tokens_context = tokens_context[:context_length - len(tokens_needle)]

        if depth_percent == 100:
            # If your depth percent is 100 (which means your needle is the last thing in the doc), throw it at the end
            tokens_new_context = tokens_context + tokens_needle
        else:
            # Go get the position (in terms of tokens) to insert your needle
            insertion_point = int(len(tokens_context) * (depth_percent / 100))
            # import ipdb; ipdb.set_trace()

            # tokens_new_context represents the tokens before the needle
            tokens_new_context = tokens_context[:insertion_point]

            # We want to make sure that we place our needle at a sentence break so we first see what token a '.' is
            if(self.model_provider in ["LLaMA", "LongLLaMA"]): period_tokens = [29889, 869]
            elif(self.model_provider == "Mistral"): period_tokens = [842, 28723]
            elif(self.model_provider == "GLM"): period_tokens = [918, 30930]
            else: period_tokens = self.encode_text_to_tokens('.')
            
            # Then we iteration backwards until we find the first period
            while tokens_new_context and tokens_new_context[-1] not in period_tokens:
                insertion_point -= 1
                tokens_new_context = tokens_context[:insertion_point]

            print_single("insertion at %d" % insertion_point)
            # Once we get there, then add in your needle, and stick the rest of your context in on the other end.
            # Now we have a needle in a haystack
            tokens_new_context += tokens_needle + tokens_context[insertion_point:]

        # Convert back to a string and return it
        new_context = self.decode_tokens(tokens_new_context)
        return new_context

    def get_context_length_in_tokens(self, context):
        if self.model_provider in ["OpenAI", "LLaMA", "Mistral", "GLM"]:
            return len(self.enc.encode(context))
        elif self.model_provider == "Anthropic":
            # Assuming you have a different encoder for Anthropic
            encoded = self.enc.encode(context)
            return len(self.enc.encode(context).ids)
        else:
            
            raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")

    def read_context_files(self):
        context = ""
        max_context_length = max(self.context_lengths)
        
        try_read_context_files = 0
        while self.get_context_length_in_tokens(context) < max_context_length:
            try_read_context_files += 1
            print_single("try_read_context_files", try_read_context_files)
            print_single("curr len:", self.get_context_length_in_tokens(context))
            
            for file in glob.glob(f"{self.haystack_dir}/*.txt"):
                with open(file, 'r') as f:
                    context += f.read()
        return context

    def get_tokens_from_context(self, context):
        if self.model_provider in ["OpenAI", "LLaMA", "Mistral", "GLM"]:
            return self.enc.encode(context)
        elif self.model_provider == "Anthropic":
            # Assuming you have a different encoder for Anthropic
            return self.enc.encode(context).ids
        else:
            raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")
        
    def decode_tokens(self, tokens, context_length=None):
        if self.model_provider in ["OpenAI", "LLaMA", "Mistral", "GLM"]:
            return self.enc.decode(tokens[:context_length])
        elif self.model_provider == "Anthropic":
            # Assuming you have a different decoder for Anthropic
            return self.enc.decode(tokens[:context_length])
        else:
            raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")

    def encode_and_trim(self, context, context_length):
        tokens = self.get_tokens_from_context(context)
        if len(tokens) > context_length:
            context = self.decode_tokens(tokens, context_length)
        return context
    
    def get_results(self):
        return self.testing_results
    
    def print_start_test_summary(self):
        print ("\n")
        print ("Starting Needle In A Haystack Testing...")
        print (f"- Model: {self.model_name}")
        print (f"- Context Lengths: {len(self.context_lengths)}, Min: {min(self.context_lengths)}, Max: {max(self.context_lengths)}")
        print (f"- Document Depths: {len(self.document_depth_percents)}, Min: {min(self.document_depth_percents)}%, Max: {max(self.document_depth_percents)}%")
        print (f"- Needle: {self.needle.strip()}")
        print ("\n\n")

    def start_test(self, args):
        if self.print_ongoing_status:
            self.print_start_test_summary()
        #asyncio.run(self.run_test())
        total_start_time = time.time()
        self.run_test(args)
        total_end_time = time.time()
        print_single("Total time:", total_end_time -total_start_time)


if __name__ == "__main__":
    # Tons of defaults set, check out the LLMNeedleHaystackTester's init for more info
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--s_len', metavar='N', type=int, help='a number')
    parser.add_argument('-e', '--e_len', metavar='N', type=int, help='a number')
    parser.add_argument('--model_path', type=str, default=None, help='path to model')
    parser.add_argument('--model_name', type=str, default=None, help='name of model')
    parser.add_argument('--model_name_suffix', type=str, default=None, help='name of model')
    parser.add_argument('--model_provider', type=str, default="LLaMA", help='which model to use')
    parser.add_argument('--api_key', type=str, default="", help='OpenAI API Key')
    
    # parser.add_argument('--use_cache', type=bool, default=False, help='use KV cache')
    parser.add_argument('--context_lengths_max', type=int, default=128000, help='max context lengths')
    parser.add_argument('--context_lengths_min', type=int, default=2048, help='min context lengths')
    parser.add_argument('--context_lengths_num_intervals', type=int, default=40, help='context_lengths_num_intervals')
    parser.add_argument('--document_depth_percent_intervals', type=int, default=10, help='document_depth_percent_intervals')
    
    parser.add_argument('--haystack_dir', type=str, default="./evaluation/needle/PaulGrahamEssays", help='path to PaulGrahamEssays')
    parser.add_argument('--result_path', type=str, default="./evaluation/needle/results", help='path to result output')
    
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--prompt_template", type=str, default="base")
    # parser.add_argument("--method", type=str, default=None, help='RoPE method in [longrope pi ntk yarn]'
    #                     )
    
    # NOTE: for cube
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--use_cube", action="store_true")
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--use_warm_up", action="store_true")
    parser.add_argument("--cube_trace", action="store_true")
    parser.add_argument("--rope_method", type=str, default="s_pi")
    parser.add_argument("--rope_tmps", type=str, default="su")
    
    from evaluation.model_loader_llama import add_args
    # parser= argparse.ArgumentParser()
    parser = add_args(parser)
    
    args = parser.parse_args()

    if(args.model_path is not None):
        assert(args.model_name is None)
        model_name = args.model_path
    else: 
        assert(args.model_name is not None)
        model_name = args.model_name

    cube.init()
    
    ht = LLMNeedleHaystackTester(model_name=model_name, 
                                 model_name_suffix=args.model_name_suffix,
                                 model_provider=args.model_provider,
                                 save_contexts=True,
                                 save_results=True,
                                 openai_api_key= args.api_key,
                                 context_lengths_min= args.context_lengths_min,
                                 context_lengths_max= args.context_lengths_max,
                                 document_depth_percent_intervals=args.document_depth_percent_intervals, 
                                 haystack_dir = args.haystack_dir,
                                 result_path = args.result_path,
                                 use_cache = args.use_cache,
                                 context_lengths_num_intervals = args.context_lengths_num_intervals,
                                 args_rope = args,
                                 prompt_template=args.prompt_template
                                 )
    if not args.cube_trace:
        ht.start_test(args)
