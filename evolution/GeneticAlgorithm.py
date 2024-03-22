import numpy as np
import logging, os
from evaluation.perplexity import compute_perplexity
from rope.LlamaLongRoPEScaledRotaryEmbedding import LlamaLongRoPEScaledRotaryEmbedding
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
            

def get_unique_filename(filename, counter=1):
    base, ext = os.path.splitext(filename)
    if counter > 1:
        filename = f"{base}_{counter}{ext}"
    while os.path.exists(filename):
        # print("exist")
        filename = f"{base}_{counter}{ext}"
        counter += 1
    return filename, counter

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
                LlamaLongRoPEScaledRotaryEmbedding(dim = 128,
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
        pass
        # # 你的变异逻辑
        # # 返回变异后的个体
        # list_step = self.list_step
        # # len_index = self.alpha_list[0 , :]
        # # curr_index = np.where(len_index == self.max_length)
        # # evo_list = self.alpha_list[1: , curr_index]
        # # 1.0 , scale+1
        # evo_list = np.arange(1.0,  self.scale + 1.0 +list_step , list_step)
        # print(f"\nevo_list range:[{evo_list.min()}, {evo_list.max()}]" )
        # while new_alpha in self.history:
        #     for p in range(new_alpha.shape[0]):
        #         # print(f'i={i}')
        #         if np.random.rand() < 0.3:
        #             if self.args.longrope_method == "increase":
        #                 if p == 0:
        #                     evo_list_curr = np.arange(1.0, evo_list.max(), list_step)
        #                 else:
        #                     evo_list_curr = np.arange(new_alpha[p-1], new_alpha[p+1]+list_step, list_step)
        #             elif self.args.longrope_method == "normal":
        #                 evo_list_curr = np.arange(1.0, evo_list.max()+list_step, list_step)
                       
        #             if evo_list_curr.shape[0] > 0:
        #                 layer_index = np.random.randint(0, evo_list_curr.shape[0])
        #                 new_alpha = new_alpha.copy()
        #                 new_alpha[p] = evo_list_curr[layer_index]
            
        # indv = [new_alpha, self.eval_ppl(new_alpha)]
        # self.history.add(indv)
        # return indv
     

    def mutate_valid_alpha(self, new_alpha):
        pass
        # # 你的变异逻辑
        # # 返回变异后的个体
        # list_step = self.list_step
        # len_index = self.alpha_list[0 , :]
        # curr_index = np.where(len_index == self.max_length)
        # evo_list = self.alpha_list[1: , curr_index]
        # # 1.0 , scale+1
        # evo_list = np.arange(1.0,  self.scale + 1.0 +list_step , list_step)
        # print(f"\nevo_list range:[{evo_list.min()}, {evo_list,max()}]" )
        # while new_alpha in self.history:
        #     for p in range(new_alpha.shape[0]):
        #         # print(f'i={i}')
        #         if np.random.rand() < 0.3:
        #             if self.args.longrope_method == "increase":
        #                 if p == 0:
        #                     evo_list_curr = np.arange(1.0, new_alpha[p+1], list_step)
        #                 elif p == new_alpha.shape[0]-1:
        #                     evo_list_curr = np.arange(new_alpha[p-1], evo_list.max()+list_step, list_step)
        #                 else:
        #                     evo_list_curr = np.arange(new_alpha[p-1], new_alpha[p+1]+list_step, list_step)
        #             elif self.args.longrope_method == "decrease":
        #                 if p == 0:
        #                     evo_list_curr = np.arange(new_alpha[p+1], evo_list.max(), list_step)
        #                 elif p == new_alpha.shape[0]-1:
        #                     evo_list_curr = np.arange(1.0, new_alpha[p-1], list_step)
        #                 else:
        #                     evo_list_curr = np.arange(new_alpha[p+1], new_alpha[p-1]+list_step, list_step)
        #             elif self.args.longrope_method == "normal":
        #                 evo_list_curr = np.arange(1.0, evo_list.max()+list_step, list_step)
                       
        #             if evo_list_curr.shape[0] > 0:
        #                 layer_index = np.random.randint(0, evo_list_curr.shape[0])
        #                 new_alpha = new_alpha.copy()
        #                 new_alpha[p] = evo_list_curr[layer_index]
            
        # indv = [new_alpha, self.eval_ppl(new_alpha)]
        # self.history.add(indv)
        # return indv
            
    def crossover_valid_alpha(self, par_alpha1, par_alpha2):
        pass
        # # 你的交叉逻辑
        # # 返回交叉后的个体
        # assert par_alpha1 in self.history and \
        #     par_alpha2 in self.history
        # assert np.array_equal(par_alpha1, par_alpha2) == False
        # par_alpha = par_alpha1.copy()
        # for _ in range(self.max_crossover_try):
        #     for i in range(par_alpha.shape[0]):
        #         if np.random.rand() < 0.3:
        #             par_alpha = par_alpha.copy()
        #             if np.random.rand() < 0.5:
        #                 par_alpha[i] = par_alpha2[i]
                        
        #             if (par_alpha in self.history):
        #                 continue
        #             if self.args.longrope_method == "increase" and (not np.all(np.diff(par_alpha)>=0)):
        #                 continue
        #             if self.args.longrope_method == "decrease" and (not np.all(np.diff(par_alpha)<=0)):
        #                 continue
                    
        #             indv = [par_alpha, self.eval_ppl(par_alpha)]
        #             self.history.add(indv)
        #             return indv
        # return None

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

        # init log
        filename_log = f"./evolution/log/{self.args.longrope_method}-{self.args.longrope_method}-{self.max_length}-step-{self.list_step}-it-{self.max_time_budget}-start_token-{self.args.start_token}.log"
        filename_log, counter_log = get_unique_filename(filename_log)

        filename_recovery = f"./evolution/log/{self.args.longrope_method}-{self.args.longrope_method}-{self.max_length}-step-{self.list_step}-it-{self.max_time_budget}-start_token-{self.args.start_token}.json"
        filename_recovery, counter_recovery = get_unique_filename(filename_recovery, counter_log)
        
        logging.basicConfig(level=logging.INFO, filename=filename_log, filemode='a+')
        log("====\n\n====\n\n====\n\n====")
        log(f"$$model: {self.args.model}\n {self.args.longrope_init_para} ")

        if self.args.recovery != None:
            # recovery from json
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

            log(f"recovery from {self.args.recovery}")
        else:
            population = []
            latest_iteration = 0
            # generate init individual
            new_alpha = self.original_alpha
            indv = [new_alpha, self.eval_ppl(new_alpha)]
            population.append(indv)
            self.history.add(indv)
            if self.verbose:
                log(f"Initial indv:{indv}")
                
            # Generate random population
            if self.verbose:
                log('=' * 100)
                log("Generate random population...")
            for i in range(self.population_size - 1):
                new_indv = self.mutate_valid_alpha(new_alpha)
                population.append(new_indv)
                if self.verbose:
                    log(f'[population {i:3d}]: {new_indv}')
        
        # generate start token
        if self.args.longrope_method == "dim_mono_n":
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
        
        # NOTE convert dim pie mono -> dim mono
        if self.args.longrope_method == "dim_mono":
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
        
        if self.verbose:
            log('=' * 100)
            log("Start Evolution...")

        best_valids = [1000]
        best_evo_alpha = None
        
        with tqdm(
            total=self.max_time_budget,
            desc="Searching",
            disable=(not self.verbose),
        ) as t:
            for i in range(latest_iteration + self.max_time_budget):
                population = [indv for indv in population if indv is not None]
                # population = list(set(population))
                parents = sorted(population, key=lambda x: x[1])[ : self.parents_size]
                
                # Save informations
                data = {'iteration': i, 'population': parents, 'history_alpha': self.history.alpha_list, 'history_ppl': self.history.ppl_list}
                json_str = json.dumps(data, indent = 4, default=lambda x: x.tolist())
                with open(filename_recovery, 'w') as file:
                    file.write(json_str)
                    
                # start token 2
                if self.args.longrope_method == "dim_mono_n":
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
                
                # Best ppl
                ppl = parents[0][1]
                t.set_postfix({"ppl": ppl})
                if self.verbose:
                    log('=' * 50)
                    log(f"[Iter {i + 1} Best]: {parents[0]}")

                if ppl < best_valids[-1]:
                    best_valids.append(ppl)
                    best_evo_alpha = parents[0]
                else:
                    best_valids.append(best_valids[-1])

                population = parents

                # Mutation
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
                    if self.args.longrope_method == "dim_mono_n":
                        # idx_indv_list = range(min(5, len(parents)))
                        # select 5 from top 10 indv
                        new_alpha, ppl = mutated_indv[0], mutated_indv[1]
                        print("start token=0:", new_alpha)
                        start_indv = self.try_start_token_mutate(new_alpha, ppl)
                        if (start_indv != None) and (start_indv[0] not in self.history) and (not self.in_population(start_indv, population)):
                            population.append(start_indv)
                            self.history.add(start_indv)
                    

                # Crossover
                for j in range(self.crossover_size):
                    idx1, idx2 = np.random.choice(self.parents_size, 2, replace=False)
                    idx1, idx2 = sorted([idx1, idx2])
                    par_alpha1 = parents[idx1][0]
                    par_alpha2 = parents[idx2][0]
                    
                    # Crossover
                    crossover_indv = self.crossover_valid_alpha(par_alpha1, par_alpha2)
                    
                    if crossover_indv is None:
                        if self.verbose:
                            log(f'Crossover reach max {self.max_crossover_try} trys. Mutate from par1 instead.')
                        crossover_indv = self.mutate_valid_alpha(par_alpha1)
                        if self.verbose:
                            log(f'*** crossover ok {j} ***')
                            log(crossover_indv)
                    population.append(crossover_indv)
                    
                     
                t.update(1)

                if self.verbose and i % 20 == 0:
                    log('-' * 50)
                    log(f'Iter {i + 1} new population')
                    for j, indv in enumerate(population):
                        log(f'[indv {j:3d}]{indv}')
                
                if i > latest_iteration and i % 20 == 0 :
                    # full output
                    output_alpha = best_evo_alpha[0]
                    
                    # save tmp result
                    filename = f"./evolution/search_result/tmp-{self.args.longrope_method}-{self.args.longrope_method}-{self.max_length}-step-{self.list_step}-it-{i}_{self.max_time_budget}.csv"
                    
                    np.savetxt(get_unique_filename(filename), output_alpha = best_evo_alpha[0], delimiter=',' )
                    
                        
                if i >= latest_iteration+20 and abs(best_valids[i] - best_valids[i-20]) < 0.001:
                    # early stop 
                    break
                

        if self.verbose:
            log('-' * 50)
            np.set_printoptions(threshold=np.inf, linewidth=np.inf)
            best_evo_alpha_full = np.array_str(best_evo_alpha[0], precision=4, suppress_small=True)
            
            log(f'[best_valids] {best_valids}')
            log(f'[best_evo_alpha] {best_evo_alpha}\n [FULL]: \n{best_evo_alpha_full}' )
        
        return best_evo_alpha
