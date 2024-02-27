import math
import torch
import numpy as np

class LlamaDynamicLongRoPEScaledRotaryEmbedding(torch.nn.Module):

    def __init__(self, dim, 
                 max_position_embeddings=4096,
                 scale=1.0,
                 base=10000, device=None, 
                 lambda_1=np.zeros((64,)), finetuned=False,
                 original_max_position_embeddings=4096,
                 mscale = 1.0,
                 tmps = "su"
                ):
        super().__init__()
        
        self.dim = dim
        self.base = base
        
        self.scale = scale
        self.modle_factor = scale
        
        self.lambda_1 = lambda_1
        self.mscale = mscale
        self.original_max_position_embeddings = original_max_position_embeddings
        
        self.tmps = tmps
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )
        
    def _get_mscale(self, scale=1):
        if scale <= 1:
            return 1.0
        # return 0.1 * math.log(scale) + 1.0
        return math.sqrt( math.log(scale*self.original_max_position_embeddings)/math.log(self.original_max_position_embeddings) )
    
    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        dim = self.dim
        base = self.base
        scaling_factor = max(seq_len / (1.0*self.original_max_position_embeddings), 1.0)
        
        base_1 = torch.from_numpy(self.lambda_1).to(device) # [dim / 2]
        assert base_1.shape[0] == dim // 2 , f"lambda_1 error : {base_1.shape[0]}"
        # print("base_1-orgin", base_1) # 128k 1.0-32.0
        
        # print(base_1)
        # dynamic:$$ 
        base_1 = 1.0 + (base_1 - 1.0) * (seq_len - self.original_max_position_embeddings)/(self.original_max_position_embeddings* (self.scale - 1))
        
        # 1-32 -> 1-64
        # 1-32 (-1)-> 0-31 (*63/62)-> 0-63 -> 1-64
        
        # dynamic 2 
        
        # base_1_old = base_1.clone()
        # for i in range(base_1.shape[0]):
        #     if base_1[i] > 2.0:
        #         base_1[i] = 1.0 + (base_1_old[i] - 1.0) * (seq_len - self.original_max_position_embeddings)/(self.original_max_position_embeddings* (self.scale - 1))
        
        #print("base after seq_len", seq_len, "self.scale", self.scale, "base_1", base_1 )
        # dim_s_pi_new = self.lambda_1.copy()
        # for dim in range(self.lambda_1.shape[0]):
        #     if self.lambda_1[dim] != 1.0:
        #         dim_s_pi_new[dim] = (self.lambda_1[dim] - 1.0) * (seq_len - self.original_max_position_embeddings)/self.original_max_position_embeddings + 1.0
        # print("seq_len", seq_len)
                 
        if self.tmps == "su":
            self.mscale = float(self._get_mscale(scaling_factor))
        else:
            self.mscale = 1.0       
        # self.mscale = float(self._get_mscale(scaling_factor))
        # print("self.mscale", self.mscale)
        
        inv_freq = 1.0 / ( base_1 * ( self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)) )
        
        self.register_buffer("inv_freq", inv_freq)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", (emb.cos() * self.mscale)[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", (emb.sin() * self.mscale)[None, None, :, :].to(dtype), persistent=False)
        
    def forward(self, x, seq_len=None):

        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
                self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
                self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            )
        

