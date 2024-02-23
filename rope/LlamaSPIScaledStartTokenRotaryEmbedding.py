import math
import torch
import numpy as np

class LlamaSPIScaledStartTokenRotaryEmbedding(torch.nn.Module):

    def __init__(self, dim, 
                 max_position_embeddings=4096,
                 scale=1.0,
                 base=10000, device=None, 
                 lambda_1=np.zeros((64,)), finetuned=False,
                 original_max_position_embeddings=4096,
                 mscale = 1.0,
                 tmps="su",
                 start_token=0,
                 cos_sin_origin=None
                ):
        super().__init__()
        
        self.dim = dim
        self.base = base
        
        self.scale = scale
        # self.ntk_factor = ntk_factor
        self.lambda_1 = lambda_1
        self.mscale = mscale
        self.original_max_position_embeddings = original_max_position_embeddings
        
        self.tmps = tmps
        self.start_token = start_token
        self.cos_sin_origin=cos_sin_origin
        
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )
        
    def _get_mscale_su(self, scale=1):
        if scale <= 1:
            return 1.0
        # return 0.1 * math.log(scale) + 1.0
        return math.sqrt( math.log(scale*self.original_max_position_embeddings)/math.log(self.original_max_position_embeddings) )
    
    def _get_mscale_yarn(self, scale=1):
        if scale <= 1:
            return 1.0
        return 0.1 * math.log(scale) + 1.0
        # return math.sqrt( math.log(scale*self.original_max_position_embeddings)/math.log(self.original_max_position_embeddings) )
    
    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        dim = self.dim
        base = self.base
        scaling_factor = max(seq_len / (1.0*self.original_max_position_embeddings), 1.0)
        
        base_1 = torch.from_numpy(self.lambda_1).to(device) # [dim / 2]
        assert base_1.shape[0] == dim // 2 , f"lambda_1 error : {base_1.shape[0]}"
        # print(base_1)
        if self.tmps == "su":
            self.mscale = float(self._get_mscale_su(scaling_factor))
        elif self.tmps == "yarn":
            self.mscale = float(self._get_mscale_yarn(scaling_factor))
        elif self.tmps == "non":
            self.mscale = 1.0
        else:
            raise ValueError(
                f"args.tmps must in [su, yarn, non]"
            )
        # print("self.mscale", self.mscale)
        
        inv_freq = 1.0 / ( base_1 * ( self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)) )
        
        self.register_buffer("inv_freq", inv_freq)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # [seq_len, dim/2]
        
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        # [seq_len, dim]
        
        emb_cos, emb_sin = (emb.cos() * self.mscale)[None, None, :, :], (emb.sin() * self.mscale)[None, None, :, :]
        # [1, 1, seq_len, dim]
        
        if self.start_token > 0:
            # inv_freq_origin = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            # freqs_origin = torch.einsum("i,j->ij", t, inv_freq_origin)
            # # [seq_len, dim/2]
            # emb_origin = torch.cat((freqs_origin, freqs_origin), dim=-1)
            assert self.cos_sin_origin != None
            emb_origin_cos, emb_origin_sin = self.cos_sin_origin
            emb_cos[:, :, 0:int(self.start_token), :] = emb_origin_cos[:, :, 0:int(self.start_token), :]
            emb_sin[:, :, 0:int(self.start_token), :] = emb_origin_sin[:, :, 0:int(self.start_token), :]
            # print(f"change{self.start_token}")
            
        self.register_buffer("cos_cached", emb_cos.to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb_sin.to(dtype), persistent=False)
        
    def forward(self, x, seq_len=None):

        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
                self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
                self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            )

