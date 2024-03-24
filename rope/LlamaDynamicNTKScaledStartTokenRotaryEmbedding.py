import math
import torch
import numpy as np

class LlamaDynamicNTKScaledStartTokenRotaryEmbedding(torch.nn.Module):

    def __init__(self, dim, 
                 max_position_embeddings=4096,
                 scale=1.0,
                 base=10000, device=None, 
                 original_max_position_embeddings=4096,
                 mscale = 1.0,
                 start_token=0,
                 cos_sin_origin=None
                ):
        super().__init__()
        
        self.dim = dim
        self.base = base
        
        self.scale = scale
        self.mscale = mscale
        self.original_max_position_embeddings = original_max_position_embeddings
        
        self.start_token = start_token
        self.cos_sin_origin=cos_sin_origin
        
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )
        
    
    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        if seq_len > self.original_max_position_embeddings:
            self.base = self.base * (
                (self.scale * seq_len / self.original_max_position_embeddings) - (self.scale - 1)
            ) ** (self.dim / (self.dim - 2))
            
        inv_freq = 1.0 / ( self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)) 
        
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
            print("start token in rope")
            assert self.cos_sin_origin != None
            emb_origin_cos, emb_origin_sin = self.cos_sin_origin
            emb_cos[:, :, 0:int(self.start_token), :] = emb_origin_cos[:, :, 0:int(self.start_token), :]
            emb_sin[:, :, 0:int(self.start_token), :] = emb_origin_sin[:, :, 0:int(self.start_token), :]
            
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

