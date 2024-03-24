
import numpy as np
from evaluation.perplexity import compute_perplexity
from rope.LlamaLinearScaledRotaryEmbedding import LlamaLinearScaledRotaryEmbedding
from evolution.GeneticAlgorithm import GeneticAlgorithm

class PIStaticGeneticAlgorithm(GeneticAlgorithm):

    def eval_ppl(self, new_alpha):
        # config_compute = [loaded, tokenizer, input_texts, tokenizer.bos_token, args.sliding_window, args.truncate, args.aggressive_memory]
        loaded, tokenizer, input_texts, tokenizer.bos_token = self.config_compute
        max_length = self.max_length
        assert isinstance(new_alpha, float), "pi static para != 1"
        
        # dim_len = 128 // 2 
        dim_alpha = new_alpha
        
        # if self.args.search_twice:
        #     dim_alpha = dim_alpha * self.lambda_1[0, :]
        for each in loaded.model.layers:
            each.self_attn.rotary_emb = \
                LlamaLinearScaledRotaryEmbedding(dim = 128,
                    scale=dim_alpha, device=each.self_attn.rotary_emb.inv_freq.device)
                
        ppl = compute_perplexity(model=loaded, tokenizer=tokenizer, encodings=input_texts,
            add_start_token=tokenizer.bos_token is not None, max_length=max_length,
            sliding_window=self.args.sliding_window, truncate=self.args.truncate,
            )['mean_perplexity']

        print(f"==: {max_length}={ppl}")
        return ppl

    
    def mutate_valid_alpha(self, new_alpha):
        list_step = self.list_step
        evo_list = np.arange(max(1.0, self.scale*0.9), (1.1*self.scale) + list_step, list_step)
        print("evo_list", evo_list)
        while (new_alpha in self.history):
            evo_list_curr = evo_list
            if evo_list_curr.shape[0] > 0:
                layer_index = np.random.randint(0, evo_list_curr.shape[0])
                print("layer_index", layer_index)
                new_alpha = evo_list_curr[layer_index]
        
        print("self.history.print()")
        self.history.print()
        print("new_alpha", new_alpha)
        indv = [new_alpha, self.eval_ppl(new_alpha)]
        self.history.add(indv)
        return indv

    def crossover_valid_alpha(self, par_alpha1, par_alpha2):
        pass
        # assert par_alpha1 in self.history and \
        #     par_alpha2 in self.history
        # assert np.array_equal(par_alpha1, par_alpha2) == False
        # par_alpha = par_alpha1.copy()
        # for _ in range(self.max_crossover_try):
            
        #     if np.random.rand() < 0.3:
        #         par_alpha = par_alpha.copy()
        #         if np.random.rand() < 0.5:
        #             par_alpha = par_alpha2.copy()
                        
        #         if (par_alpha in self.history):
        #             continue
                
        #         indv = [par_alpha, self.eval_ppl(par_alpha)]
        #         self.history.add(indv)
        #         return indv
        # return None
