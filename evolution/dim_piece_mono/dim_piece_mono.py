import numpy as np
from evaluation.perplexity import compute_perplexity
from rope.LlamaSPIScaledRotaryEmbedding import LlamaSPIScaledRotaryEmbedding
from evolution.GeneticAlgorithm import GeneticAlgorithm


class DimPieceMonoGeneticAlgorithm(GeneticAlgorithm):

    def eval_ppl(self, new_alpha, if_slide=False, input_texts=None):
        loaded, tokenizer, input_texts, tokenizer.bos_token = self.config_compute    
        max_length = self.max_length
        assert new_alpha.shape[0] == 2+64, f"Piece Mono para{new_alpha.shape[0]} != 2+64"
        
        dim_len = 128 // 2 
        dim_alpha = new_alpha[2:]
        
        if self.args.search_twice:
            dim_alpha = dim_alpha * self.lambda_1[0, :]
            
        for each in loaded.model.layers:
            each.self_attn.rotary_emb = \
                LlamaSPIScaledRotaryEmbedding(dim = 128,
                    lambda_1=dim_alpha)
  
        
        ppl = compute_perplexity(model=loaded, tokenizer=tokenizer, encodings=input_texts,
            add_start_token=tokenizer.bos_token is not None, max_length=max_length,
            sliding_window=self.args.sliding_window, truncate=self.args.truncate,
            aggressive_memory=self.args.aggressive_memory)['mean_perplexity']
        print(f"==: {max_length}={ppl}")
        return ppl

    def mutate_valid_alpha(self, new_alpha):
        list_step = self.list_step
        evo_list = np.arange(1.0, 1.0 + self.scale + list_step , list_step)
        
        while (new_alpha in self.history):
            if np.random.rand() < 0.3:
                alpha_dim = new_alpha[0]
                beta_dim = new_alpha[1]
                std_dev = 2.0
                
                if np.random.rand() < 0.5:
                    alpha_dim = int(np.random.normal(alpha_dim, std_dev))
                else:
                    beta_dim = int(np.random.normal(beta_dim, std_dev))
                    
                cnt = 0
                while not (0 <= alpha_dim and alpha_dim < beta_dim and beta_dim <= 63):
                    if cnt > 10000: break
                    if np.random.rand() < 0.5:
                        alpha_dim = int(np.random.normal(alpha_dim, std_dev)) * 1.0
                    else:
                        beta_dim = int(np.random.normal(beta_dim, std_dev)) * 1.0
                          
                    cnt += 1
                
                new_alpha = new_alpha.copy()
                new_alpha[0] = alpha_dim
                new_alpha[1] = beta_dim
                
                # new dim 64
                scale = self.max_length / self.args.original_max_position_embeddings
                if not self.args.small_extra:
                    for dim in range(2, new_alpha.shape[0]-2):
                        if dim <= alpha_dim + 2: 
                            new_alpha[dim] = 1.0
                        elif dim > beta_dim + 2:
                            new_alpha[dim] = scale
                        else:
                            if dim == 2 + 0:
                                evo_list_curr = np.arange(1.0, new_alpha[dim+1], list_step)
                            elif dim == 2 + new_alpha.shape[0]-1:
                                evo_list_curr = np.arange(new_alpha[dim-1], evo_list.max()+list_step, list_step)
                            else:
                                evo_list_curr = np.arange(new_alpha[dim-1], new_alpha[dim+1]+list_step, list_step)
                                
                            if evo_list_curr.shape[0] > 0:
                                layer_index = np.random.randint(0, evo_list_curr.shape[0])
                                new_alpha = new_alpha.copy()
                                new_alpha[dim] = evo_list_curr[layer_index]
                else:
                    evo_list = np.arange(0.6, 1.0 + self.scale + list_step , list_step)
                    for dim in range(2, new_alpha.shape[0]):
                        if dim < alpha_dim + 2: 
                            new_alpha[dim] = new_alpha[int(alpha_dim + 2)]
                        elif dim > beta_dim + 2:
                            new_alpha[dim] = new_alpha[int(beta_dim + 2)]
                        else:
                            if dim == 2 + alpha_dim:
                                evo_list_curr = np.arange(evo_list.min(), new_alpha[dim+1], list_step)
                            elif dim == 2 + beta_dim:
                                evo_list_curr = np.arange(new_alpha[dim-1], evo_list.max()+list_step, list_step)
                            else:
                                evo_list_curr = np.arange(new_alpha[dim-1], new_alpha[dim+1]+list_step, list_step)
                                
                            if evo_list_curr.shape[0] > 0:
                                layer_index = np.random.randint(0, evo_list_curr.shape[0])
                                new_alpha = new_alpha.copy()
                                new_alpha[dim] = evo_list_curr[layer_index]
                    new_alpha[2: int(2 + alpha_dim)] = new_alpha[int(alpha_dim + 2)]
                           
        indv = [new_alpha, self.eval_ppl(new_alpha)]
        self.history.add(indv)
        return indv

    def crossover_valid_alpha(self, par_alpha1, par_alpha2):
        assert par_alpha1 in self.history and \
            par_alpha2 in self.history
        assert np.array_equal(par_alpha1, par_alpha2) == False
        par_alpha = par_alpha1.copy()
        for _ in range(self.max_crossover_try):
            for i in range(par_alpha.shape[0]):
                if np.random.rand() < 0.3:
                    
                    par_alpha = par_alpha.copy()
                    if np.random.rand() < 0.5:
                        if i < 2:
                            par_alpha[0] = par_alpha2[0]
                            par_alpha[1] = par_alpha2[1]
                        else:
                            par_alpha[i] = par_alpha2[i]
                            
                    if (par_alpha in self.history):
                        continue
                    
                    if par_alpha[0] > par_alpha[1]:
                        continue
                    if (not np.all(np.diff(par_alpha[2:])>=0)):
                        continue
                    
                    indv = [par_alpha, self.eval_ppl(par_alpha)]
                    self.history.add(indv)
                    return indv
        return None

