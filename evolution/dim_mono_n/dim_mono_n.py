
import numpy as np
from evaluation.perplexity import compute_perplexity
from rope.LlamaLongRoPEScaledRotaryEmbedding import LlamaLongRoPEScaledRotaryEmbedding
from rope.LlamaLongRoPEScaledStartTokenRotaryEmbedding import  LlamaLongRoPEScaledStartTokenRotaryEmbedding
from evolution.GeneticAlgorithm import GeneticAlgorithm


class DimMonoNGeneticAlgorithm(GeneticAlgorithm):

    def eval_ppl(self, new_alpha):
        # config_compute = [loaded, tokenizer, input_texts, tokenizer.bos_token, args.sliding_window, args.truncate, args.aggressive_memory]
        loaded, tokenizer, input_texts, tokenizer.bos_token = self.config_compute
        max_length = self.max_length
        assert new_alpha.shape[0] == 64+1, f"Dim Mono N para {new_alpha.shape[0] } != 64+1"
        
        # start_tokens
        start_token = new_alpha[0]
        dim_len = 128 // 2 
        dim_alpha = new_alpha[1:]
        
        if start_token > 0:
            print("start_token", start_token)

        for each in loaded.model.layers:
            each.self_attn.rotary_emb = \
                LlamaLongRoPEScaledStartTokenRotaryEmbedding(
                    dim = each.self_attn.head_dim,
                    lambda_1=dim_alpha, start_token=start_token,
                    cos_sin_origin=self.cos_sin_origin
                    )
        
        # config_compute = [loaded, tokenizer, input_texts, tokenizer.bos_token, args.sliding_window, args.truncate, args.aggressive_memory]
        ppl = compute_perplexity(model=loaded, tokenizer=tokenizer, encodings=input_texts,
            add_start_token=tokenizer.bos_token is not None, max_length=max_length,
            sliding_window=self.args.sliding_window, truncate=self.args.truncate,
            )['mean_perplexity']

        print(f"==: {max_length}={ppl}")
        return ppl


    def mutate_valid_alpha(self, new_alpha):
        # 你的变异逻辑
        # 返回变异后的个体
        list_step = self.list_step
        evo_list = np.arange(1.0, 1.0 + self.scale + list_step , list_step)
        
        while (new_alpha in self.history) or (not np.all(np.diff(new_alpha[1:])>=0)):

            for dim in range(1, new_alpha.shape[0]-1):
                if np.random.rand() < 0.3:
                    if dim == 0+1:
                        evo_list_curr = np.arange(1.0, new_alpha[dim+1], list_step)
                    elif dim == new_alpha.shape[0]-1+1:
                        evo_list_curr = np.arange(new_alpha[dim-1], evo_list.max()+list_step, list_step)
                    elif dim > 0+1:
                        evo_list_curr = np.arange(new_alpha[dim-1], new_alpha[dim+1]+list_step, list_step)
                        
                    if evo_list_curr.shape[0] > 0:
                        layer_index = np.random.randint(0, evo_list_curr.shape[0])
                        new_alpha = new_alpha.copy()
                        new_alpha[dim] = evo_list_curr[layer_index]
                        
        indv = [new_alpha, self.eval_ppl(new_alpha)]
        self.history.add(indv)
        return indv
       
    def crossover_valid_alpha(self, par_alpha1, par_alpha2):
        assert par_alpha1 in self.history and \
            par_alpha2 in self.history
        # assert np.array_equal(par_alpha1, par_alpha2) == False, f"{par_alpha1, par_alpha2}"
        if np.array_equal(par_alpha1, par_alpha2) == True:
            return "remove"
        
        par_alpha = par_alpha1.copy()
        for _ in range(self.max_crossover_try):
            for i in range(par_alpha.shape[0]):
                if np.random.rand() < 0.3:
                    
                    par_alpha = par_alpha.copy()
                    if np.random.rand() < 0.5:
                        if i < 1:
                            par_alpha[0] = par_alpha2[0]
                        else:
                            par_alpha[i] = par_alpha2[i]
                            
                    if (par_alpha in self.history):
                        continue
                    
                    if (not np.all(np.diff(par_alpha[1:])>=0)):
                        continue
                    
                    indv = [par_alpha, self.eval_ppl(par_alpha)]
                    self.history.add(indv)
                    return indv
        return None
