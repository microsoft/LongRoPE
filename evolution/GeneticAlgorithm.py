import numpy as np
import logging, os
from evaluation.perplexity import compute_perplexity
from rope.LlamaSPIScaledRotaryEmbedding import LlamaSPIScaledRotaryEmbedding
from tqdm import tqdm
import json
import torch

from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

def log(text):
    try:
        logging.info(text)
        print(text)  
    except IOError as e:
        if e.errno == errno.EPIPE:
            pass
        

class History:
    def __init__(self) -> None:
        self.alpha_list = []
        self.ppl_list = []
        
    def clear(self):
        self.alpha_list = []
        self.ppl_list = []

    def __contains__(self, other):
        if isinstance(other, np.ndarray):
            other_str = "_".join(map(str, other))
            for item in self.alpha_list:
                item_str = "_".join(map(str, item))
                if item_str == other_str:
                    return True
            return False

    def add(self, indv):
        if not self.__contains__(indv[0]):
            self.alpha_list.append(indv[0])
            self.ppl_list.append(indv[1])

    def print(self):
        for alpha in self.alpha_list:
            print('history,',alpha)
            

def get_unique_filename(filename):
    counter = 1
    base, ext = os.path.splitext(filename)
    while os.path.exists(filename):
        # base, ext = os.path.splitext(filename)
        print("exist")
        # 文件已存在，添加后缀
        filename = f"{base}_{counter}{ext}"
        counter += 1
    return filename

# TODO: indv class / structure
class GeneticAlgorithm:
    def __init__(self, args, config_compute, max_length, 
                 loaded_parameters,
                 verbose = True,
                 init_alpha=None,
                lambda_1=None):
        # config_compute = [loaded, tokenizer, input_texts, tokenizer.bos_token, args.sliding_window, args.truncate, args.aggressive_memory]
        
        self.args = args
        self.config_compute = config_compute
        self.max_length = max_length
        self.scale = self.args.factor
        self.original_alpha = init_alpha
        
        self.history = History()
        
        self.verbose = verbose
        
        evo_scale = loaded_parameters["evo_scale"]
        self.population_size = int(evo_scale * loaded_parameters["population_size"])
        self.max_time_budget = int(evo_scale * loaded_parameters["max_time_budget"])
        self.mutation_numbers = int(evo_scale * loaded_parameters["mutation_numbers"])
        self.crossover_size = int(evo_scale * loaded_parameters["crossover_size"])
        self.max_crossover_try = int(evo_scale * loaded_parameters["max_crossover_try"])
        self.parents_size = int(evo_scale * loaded_parameters["parents_size"])
        self.list_step = loaded_parameters["list_step"]

        # start token prepare
        loaded, _, _, _ = self.config_compute
        seq_len = self.max_length
        tmp_device = "cpu"
        rotary_emb_origin = LlamaRotaryEmbedding(dim=loaded.model.layers[0].self_attn.head_dim, max_position_embeddings=seq_len, device=tmp_device)
        input_x = torch.zeros((1,),dtype=torch.float16, device=tmp_device)
        self.cos_sin_origin = rotary_emb_origin.forward(x=input_x, seq_len=seq_len)
        
        # twice search prepare
        # lambda_1 = np.loadtxt(open(args.s_pi_para, "rb"), delimiter=",", skiprows=0)
        assert lambda_1.shape == (32, 64), f"lambda_1 shape error {lambda_1.shape}"
        self.lambda_1 = lambda_1
            
    def eval_ppl(self, new_alpha):
        loaded, tokenizer, input_texts, tokenizer.bos_token = self.config_compute
        max_length = self.max_length
        k = 0
        for each in loaded.model.layers:
            curr_alpha = new_alpha[k]
            assert isinstance(curr_alpha, float), "not float"
            each.self_attn.rotary_emb = \
                LlamaSPIScaledRotaryEmbedding(dim = 128,
                    alpha=curr_alpha)
                # 1 row is static parameter
            k = k + 1
            
        
        ppl = compute_perplexity(model=loaded, tokenizer=tokenizer, encodings=input_texts,
            add_start_token=tokenizer.bos_token is not None, max_length=max_length,
            sliding_window=self.args.sliding_window, truncate=self.args.truncate,
            aggressive_memory=self.args.aggressive_memory, hide_progress=self.args.hide_progress)['mean_perplexity']
            
    
        print(f"==: {max_length}={ppl}")
        return ppl

    def sample_valid_alpha(self, new_alpha):
        # 你的变异逻辑
        # 返回变异后的个体
        list_step = self.list_step
        # len_index = self.alpha_list[0 , :]
        # curr_index = np.where(len_index == self.max_length)
        # evo_list = self.alpha_list[1: , curr_index]
        # 1.0 , scale+1
        evo_list = np.arange(1.0,  self.scale + 1.0 +list_step , list_step)
        print(f"\nevo_list range:[{evo_list.min()}, {evo_list.max()}]" )
        while new_alpha in self.history:
            for p in range(new_alpha.shape[0]):
                # print(f'i={i}')
                if np.random.rand() < 0.3:
                    if self.args.s_pi_method == "increase":
                        if p == 0:
                            evo_list_curr = np.arange(1.0, evo_list.max(), list_step)
                        else:
                            evo_list_curr = np.arange(new_alpha[p-1], new_alpha[p+1]+list_step, list_step)
                    elif self.args.s_pi_method == "normal":
                        evo_list_curr = np.arange(1.0, evo_list.max()+list_step, list_step)
                       
                    if evo_list_curr.shape[0] > 0:
                        layer_index = np.random.randint(0, evo_list_curr.shape[0])
                        new_alpha = new_alpha.copy()
                        new_alpha[p] = evo_list_curr[layer_index]
            
        indv = [new_alpha, self.eval_ppl(new_alpha)]
        self.history.add(indv)
        return indv
     

    def mutate_valid_alpha(self, new_alpha):
        # 你的变异逻辑
        # 返回变异后的个体
        list_step = self.list_step
        len_index = self.alpha_list[0 , :]
        curr_index = np.where(len_index == self.max_length)
        evo_list = self.alpha_list[1: , curr_index]
        # 1.0 , scale+1
        evo_list = np.arange(1.0,  self.scale + 1.0 +list_step , list_step)
        print(f"\nevo_list range:[{evo_list.min()}, {evo_list,max()}]" )
        while new_alpha in self.history:
            for p in range(new_alpha.shape[0]):
                # print(f'i={i}')
                if np.random.rand() < 0.3:
                    if self.args.s_pi_method == "increase":
                        if p == 0:
                            evo_list_curr = np.arange(1.0, new_alpha[p+1], list_step)
                        elif p == new_alpha.shape[0]-1:
                            evo_list_curr = np.arange(new_alpha[p-1], evo_list.max()+list_step, list_step)
                        else:
                            evo_list_curr = np.arange(new_alpha[p-1], new_alpha[p+1]+list_step, list_step)
                    elif self.args.s_pi_method == "decrease":
                        if p == 0:
                            evo_list_curr = np.arange(new_alpha[p+1], evo_list.max(), list_step)
                        elif p == new_alpha.shape[0]-1:
                            evo_list_curr = np.arange(1.0, new_alpha[p-1], list_step)
                        else:
                            evo_list_curr = np.arange(new_alpha[p+1], new_alpha[p-1]+list_step, list_step)
                    elif self.args.s_pi_method == "normal":
                        evo_list_curr = np.arange(1.0, evo_list.max()+list_step, list_step)
                       
                    if evo_list_curr.shape[0] > 0:
                        layer_index = np.random.randint(0, evo_list_curr.shape[0])
                        new_alpha = new_alpha.copy()
                        new_alpha[p] = evo_list_curr[layer_index]
            
        indv = [new_alpha, self.eval_ppl(new_alpha)]
        self.history.add(indv)
        return indv
            
    def crossover_valid_alpha(self, par_alpha1, par_alpha2):
        # 你的交叉逻辑
        # 返回交叉后的个体
        assert par_alpha1 in self.history and \
            par_alpha2 in self.history
        assert np.array_equal(par_alpha1, par_alpha2) == False
        par_alpha = par_alpha1.copy()
        for _ in range(self.max_crossover_try):
            for i in range(par_alpha.shape[0]):
                if np.random.rand() < 0.3:
                    par_alpha = par_alpha.copy()
                    if np.random.rand() < 0.5:
                        par_alpha[i] = par_alpha2[i]
                        
                    if (par_alpha in self.history):
                        continue
                    if self.args.s_pi_method == "increase" and (not np.all(np.diff(par_alpha)>=0)):
                        continue
                    if self.args.s_pi_method == "decrease" and (not np.all(np.diff(par_alpha)<=0)):
                        continue
                    
                    indv = [par_alpha, self.eval_ppl(par_alpha)]
                    self.history.add(indv)
                    return indv
        return None

    def try_start_token(self, new_alpha, ppl):
        # ppl = self.eval_ppl(new_alpha)
        ppl_start_token_list = {}
        print("curr ppl", ppl)
        evo_list_curr = np.array([1,2,4,8,12,16,20,24,28,32,48,64])
        # random_values = np.random.choice(evo_list_curr, size=5, replace=False)
        random_values = evo_list_curr
        for i in range(random_values.shape[0]):   
            layer_index = i
            new_alpha_start_token = new_alpha.copy()
            new_alpha_start_token[0] = random_values[layer_index]
            ppl_start_token_list[new_alpha_start_token[0]] = self.eval_ppl(new_alpha_start_token)
        
        log(ppl_start_token_list)
        
        if ppl_start_token_list != {}:
            min_ppl = min(ppl_start_token_list.values())
            if min_ppl < ppl:
                min_start_token = min(ppl_start_token_list, key=ppl_start_token_list.get)
                new_alpha_start_token[0] = min_start_token
                new_alpha = new_alpha_start_token.copy()
                log("added start token ##")
            log(f"{min_ppl} < {ppl} : diff {min_ppl - ppl}" )

        return [new_alpha, min_ppl]
    
    def try_start_token_mutate(self, new_alpha, ppl):
        # ppl = self.eval_ppl(new_alpha)
        ppl_start_token_list = {}
        print("curr ppl", ppl)
        evo_list_curr = np.array([1,2,4,8,12,16,20,24,28,32,48,64])
        random_values = np.random.choice(evo_list_curr, size=5, replace=False)
        
        # random_values = evo_list_curr
        for i in range(random_values.shape[0]):   
            layer_index = i
            new_alpha_start_token = new_alpha.copy()
            new_alpha_start_token[0] = random_values[layer_index]
            ppl_start_token_list[new_alpha_start_token[0]] = self.eval_ppl(new_alpha_start_token)
        
        log(ppl_start_token_list)
        
        if ppl_start_token_list != {}:
            min_ppl = min(ppl_start_token_list.values())
            if min_ppl < ppl:
                min_start_token = min(ppl_start_token_list, key=ppl_start_token_list.get)
                new_alpha_start_token[0] = min_start_token
                new_alpha = new_alpha_start_token.copy()
                log("added start token ##")
            log(f"{min_ppl} < {ppl} : diff {min_ppl - ppl}" )

        return [new_alpha, min_ppl]
    
    def in_population(self, indv, population):
        indv_0_str = "_".join(map(str, indv[0]))
        for indv_po_0, _ in population:
            indv_po_0 = "_".join(map(str, indv_po_0)) 
            if indv_0_str == indv_po_0:
                return True
        return False

    def run_genetic_algorithm(self):
        # 遗传算法主循环
        # population_size = 128  # 设置种群大小
        # max_time_budget = 80  # 设置时间预算
        # mutation_numbers = 16  # 设置变异操作数量
        # crossover_size = 16  # 设置交叉操作数量
        # max_crossover_try = 40

        # verbose = True  # 是否输出详细信息
        # list_step = 0.01

        if self.verbose:
            # model name:
            model_path = self.args.model[0][0]
            parts = model_path.split("/")
            # 取倒数第二个和最后一个部分，并拼接在一起
            model_name = "-".join(parts[-3:])

            # 初始化日志记录等
            filename_recovery = f"./evolution/{self.args.s_pi_method}/log/" + str(self.args.s_pi_method) + str(self.max_length) + "-step" + str(self.list_step) + \
                "-it" + str(self.max_time_budget) + "stream-"+str(self.args.stream) + model_name + ".json"
            filename_recovery = get_unique_filename(filename_recovery)
            
            filename = f"./evolution/{self.args.s_pi_method}/log/" + str(self.args.s_pi_method) + str(self.max_length) + "-step" + str(self.list_step) + \
                "-it" + str(self.max_time_budget) + "stream-"+str(self.args.stream) + model_name + ".log"

            logging.basicConfig(level=logging.INFO, filename=get_unique_filename(filename), filemode='a+')
            log("====\n\n====\n\n====\n\n====")
            log(f"$$model: {self.args.model}\n {self.args.s_pi_init_para} ")

        # evo_list = self.evo_list
        if self.args.recovery != None:
            with open(self.args.recovery, 'r') as file:
                data = json.load(file)
                population = data['population']
                population = [[np.array(sublist[0]), sublist[1]] for sublist in population]
                latest_iteration = data['iteration']
                
                if "history_alpha" in data:
                    self.history.alpha_list, self.history.ppl_list = data['history_alpha'], data['history_ppl']
                else:
                    self.history.alpha_list = [alpha[0] for alpha in population]
                    self.history.ppl_list = [alpha[1] for alpha in population]
                log(f"latest_iteration$${latest_iteration}\n {population}")
                # print(population, self.history.alpha_list)
                    
                
            print(f"recovery from {self.args.recovery}")
        else:
            population = []
            latest_iteration = 0
        
        best_valids = [1000]
        best_evol_alpha = None

        if self.verbose:
            # 输出初始化种群信息
            log('=' * 100)
            log("Generate random population...")
            
        new_alpha = self.original_alpha
        indv = [new_alpha, self.eval_ppl(new_alpha)]
        population.append(indv)
        self.history.add(indv)

        if self.verbose:
            # 输出初始个体信息
            log(f"Initial indv:{indv}")
        if self.args.recovery == None :
            for i in range(self.population_size - 1):
                # 生成种群
                if self.args.s_pi_method in ["layerwise_dim_mono", "layerwise_dim_piece", "layerwise_dim_piece_mono"]:
                    new_indv = self.sample_valid_alpha(new_alpha)
                else:
                    new_indv = self.mutate_valid_alpha(new_alpha)
                population.append(new_indv)

                if self.verbose:
                    # 输出生成的个体信息
                    log(f'[population {i:3d}]: {new_indv}')
        
        # start token 1
        if self.args.s_pi_method in [ "dim_mono_n", "dim_piece_mono_n" ] :
            # input check
            
            for i in range(len(population)):
                tmp = population[i][0]
                if tmp.shape == (64,): # dim mono n
                    tmp = np.concatenate(([0.0], tmp), axis=0) 
                    population[i][0] = tmp
                assert population[i][0].shape == (64+1,), f"shape error {population[-1][0].shape}"
            
            # mono check: from dim mono
            if "dim_mono" in self.args.recovery: 
                self.history.clear()
                for i in range(len(population)):
                    tmp = population[i][0]
                    if tmp.shape != (64+1,):
                        alpha = tmp[0,:] # first row
                    else:
                        alpha = tmp
                    for p in range(1, alpha.shape[0]-1):
                        if alpha[p] > alpha[p+1]:
                            alpha[p+1] = alpha[p]
                    flag = np.all(np.diff(alpha[1:])>=0)
                    # , f"Not mono: {alpha}"
                    log(f"{i}:{flag}, {alpha}")
                    
                    if alpha not in self.history:
                        population[i][0] = alpha
                        population[i][1] = self.eval_ppl(alpha)
                        self.history.alpha_list.append(population[i][0])
                        self.history.ppl_list.append(population[i][1])
            
            population = sorted(population, key=lambda x: x[1])[ : self.parents_size]
            indv_size = min(10, self.parents_size)
            idx_indv_list = np.random.choice([i for i in range(indv_size)], size=int(indv_size/2), replace=False).tolist()
            # select 5 from top 10 indv
            for k in idx_indv_list:
                new_alpha, ppl = population[k][0], population[k][1]   
                print("new_alpha", new_alpha)   
                start_indv = self.try_start_token(new_alpha, ppl)
                print("debug", start_indv, population)
                if (start_indv != None) and (start_indv[0] not in self.history) and (not self.in_population(start_indv, population)):
                    population.append(start_indv)
                    self.history.add(start_indv)
                    if self.verbose:
                        log(f'*** start token {k} ***')
                        log(f"start token=0: {new_alpha}")
                        log(population[k])
                        log(start_indv)
        
        if self.args.s_pi_method == "layerwise_dim_mono_non_layer" and self.args.recovery != None:
            
            print("dim mono n -> layerwise")
            # dim mono n -> layerwise
            population = sorted(population, key=lambda x: x[1])[ : min(64,len(population))]
            # (64+,) -> (64,)
            for i in range(len(population)):
                tmp = population[i][0]
                if isinstance(tmp, list):
                    tmp = np.array(tmp)
                if tmp.shape == (32, 64):
                    population[i][0] = tmp[0,:]
                elif tmp.shape == (64+2,): # dim piece mono
                    population[i][0] = tmp[2:]
                elif tmp.shape == (64+1,): # dim mono n
                    population[i][0] = tmp[1:]
                else:
                    population[i][0] = tmp
                assert population[i][0].shape == (64,), f"shape error {population[i][0].shape}"
            
            # # history reshape
            # for i in range(len(self.history.alpha_list)):
            #     tmp = self.history.alpha_list[i]
            #     if isinstance(tmp, list):
            #         tmp = np.array(tmp)
            #     if tmp.shape[0] == 64+2: 
            #         self.history.alpha_list[i] = tmp[2:]
            #     else:
            #         self.history.alpha_list[i] = tmp
            #     assert self.history.alpha_list[i].shape == (64,) ,  f"shape error {self.history.alpha_list[i].shape}"
                
            ##########################workaround for non-dim-mono-n results###################################
            
            # self.history.clear()
            for i in range(len(population)):
                tmp = population[i][0]
                if tmp.shape != (64,):
                    alpha = tmp[0,:] # first row
                else:
                    alpha = tmp
                alpha_old = alpha.copy()
                for p in range(alpha.shape[0]-1):
                    if alpha[p] > alpha[p+1]:
                        alpha[p+1] = alpha[p]
                flag = np.all(np.diff(alpha)>=0)
                if_change_ppl = not np.array_equal(alpha_old, alpha)
                # , f"Not mono: {alpha}"
                log(f"{i}:{flag}, {alpha}, {if_change_ppl}")
                
                # (64,) -> (32,64)
                alpha = np.tile(alpha, (32, 1))
                assert alpha.shape == (32,64), f"shape error {alpha}"
                if alpha not in self.history:
                    population[i][0] = alpha
                    if if_change_ppl:
                        population[i][1] = self.eval_ppl(alpha)
                    self.history.alpha_list.append(population[i][0])
                    self.history.ppl_list.append(population[i][1])
            
            
            print(population)
            print('here1',population)
            print('here2',self.history.print())

        if self.args.s_pi_method == "dim_mono_twice":
            print("convert dim pie mono -> dim mono")
            population = sorted(population, key=lambda x: x[1])[ : min(64,len(population))]
            # (64+2,) -> (64,)
            for i in range(len(population)):
                tmp = population[i][0]
                if tmp.shape == (64+2,): 
                    population[i][0] = tmp[2:]
                assert population[i][0].shape == (64,), f"shape error {population[-1][0].shape}"
            
            # history reshape
            for i in range(len(self.history.alpha_list)):
                tmp = self.history.alpha_list[i]
                # list -> np
                if isinstance(tmp, list):
                    tmp = np.array(tmp)
                # (64+2,) -> (64,)    
                if tmp.shape[0] == 64+2: 
                    self.history.alpha_list[i] = tmp[2:]
                else:
                    self.history.alpha_list[i] = tmp
                assert self.history.alpha_list[i].shape == (64,) ,  f"shape error {self.history.alpha_list[i].shape}"
        
        # if self.args.s_pi_method == "dim_mono_twice":
        #     new_alpha = np.full((64,), 1.0 )
        #     indv = [new_alpha, self.eval_ppl(new_alpha)]
        #     print("One indv:", indv )
        #     population.append(indv)
        #     self.history.add(indv)
        
        if self.verbose:
            # 输出进化开始信息
            log('=' * 100)
            log("Start Evolution...")

        with tqdm(
            total=self.max_time_budget,
            desc="Searching",
            disable=(not self.verbose),
        ) as t:
            for i in range(latest_iteration + self.max_time_budget):
                population = [indv for indv in population if indv is not None]
                parents = sorted(population, key=lambda x: x[1])[ : self.parents_size]
                
                 # 存储数据到 JSON 文件
                data = {'iteration': i, 'population': parents, 'history_alpha': self.history.alpha_list, 'history_ppl': self.history.ppl_list}
                json_str = json.dumps(data, indent = 4, default=lambda x: x.tolist())
                with open(filename_recovery, 'w') as file:
                    file.write(json_str)
                    
                # start token 2
                if self.args.s_pi_method in [ "dim_mono_n", "dim_piece_mono_n" ] :
                    idx_indv_list = range(min(5, len(parents)))
                    # select 5 from top 10 indv
                    for k in idx_indv_list:
                        if parents[k][0] in self.history:
                            continue
                        new_alpha, ppl = parents[k][0], parents[k][1]
                        print("start token=0:", new_alpha)
                        start_indv = self.try_start_token(new_alpha, ppl)
                        if (start_indv != None) and (start_indv[0] not in self.history) and (not self.in_population(start_indv, population)):
                            parents.append(start_indv)
                            self.history.add(start_indv)
                            if self.verbose:
                                log(f'*** start {k} ***')
                                log(parents[k])
                                log(start_indv)
                                
                    parents = sorted(parents, key=lambda x: x[1])[ : self.parents_size]   
                     
                ppl = parents[0][1]
                t.set_postfix({"ppl": ppl})

                if self.verbose:
                    # 输出每一代的最佳个体信息
                    log('=' * 50)
                    log(f"[Iter {i + 1} Best]: {parents[0]}")

                if ppl < best_valids[-1]:
                    best_valids.append(ppl)
                    best_evol_alpha = parents[0]
                else:
                    best_valids.append(best_valids[-1])

                population = parents

                # 变异
                for j in range(self.mutation_numbers):
                    idx = np.random.randint(self.parents_size)
                    par_alpha = parents[idx][0]
                    mutated_indv = self.mutate_valid_alpha(par_alpha)
                    population.append(mutated_indv)
                    if self.verbose:
                        log(f'*** mutate {j} ***')
                        log(parents[idx])
                        log(mutated_indv)
                        
                    # start token 3
                    if self.args.s_pi_method in [ "dim_mono_n", "dim_piece_mono_n" ] :
                        # idx_indv_list = range(min(5, len(parents)))
                        # select 5 from top 10 indv
                        # for k in idx_indv_list:
                        # k=0
                        new_alpha, ppl = mutated_indv[0], mutated_indv[1]
                        print("start token=0:", new_alpha)
                        start_indv = self.try_start_token_mutate(new_alpha, ppl)
                        if (start_indv != None) and (start_indv[0] not in self.history) and (not self.in_population(start_indv, population)):
                            population.append(start_indv)
                            self.history.add(start_indv)
                    

                # 交叉
                for j in range(self.crossover_size):
                    idx1, idx2 = np.random.choice(self.parents_size, 2, replace=False)
                    idx1, idx2 = sorted([idx1, idx2])
                    par_alpha1 = parents[idx1][0]
                    par_alpha2 = parents[idx2][0]
                    
                    # Crossover
                    crossover_indv = self.crossover_valid_alpha(par_alpha1, par_alpha2)
                    if crossover_indv is "remove":
                        # idx_to_remove = np.all(population == population[idx2], axis=0)
                        del population[idx2]
                        continue
                    
                    if crossover_indv is None:
                        if self.verbose:
                            log(f'Crossover reach max {self.max_crossover_try} trys. Mutate from par1 instead.')
                        crossover_indv = self.mutate_valid_alpha(par_alpha1)
                        if self.verbose:
                            log(f'*** crossover ok {j} ***')
                            log(crossover_indv)
                    population.append(crossover_indv)
                    
                     
                t.update(1)

                if self.verbose and i >= self.max_time_budget-2:
                    # 输出每一代的种群信息
                    log('-' * 50)
                    log(f'Iter {i + 1} new population')
                    for j, indv in enumerate(population):
                        log(f'[indv {j:3d}]{indv}')
                
                if i > latest_iteration  and i + latest_iteration % 20 == 0 :
                    # 完整输出
                    out_alpha = best_evol_alpha[0]
                    
                    filename = f"./evolution/{self.args.s_pi_method}/result_alpha/tmp" + str(self.args.s_pi_method) + str(self.max_length) + "-step" + str(self.list_step) + \
                    "-it" + f"{i}_{self.max_time_budget}" + ".csv"
                    np.savetxt(get_unique_filename(filename), out_alpha, delimiter=',' )
                    
                        
                if i >= latest_iteration+20 and abs(best_valids[i] - best_valids[i-20]) < 0.001:
                    break
                

        if self.verbose:
            # 输出最终结果信息
            log('-' * 50)
            # 完整输出
            np.set_printoptions(threshold=np.inf, linewidth=np.inf)
            best_evol_alpha_full = np.array_str(best_evol_alpha[0], precision=4, suppress_small=True)
            
            # print(best_evol_alpha_full)
            log(f'[best_valids] {best_valids}')
            log(f'[best_evol_alpha] {best_evol_alpha}\n [FULL]: \n{best_evol_alpha_full}' )
        
        return best_evol_alpha
        print("==###########===\n", best_valids, best_evol_alpha)
