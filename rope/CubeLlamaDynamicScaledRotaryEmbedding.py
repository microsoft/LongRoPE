import math
import cube
import torch

def find_correction_dim(num_rotations, dim, base=10000, max_position_embeddings=2048):
    return (dim * math.log(max_position_embeddings/(num_rotations * 2 * math.pi)))/(2 * math.log(base))

# Find dim range bounds based on rotations
def find_correction_range(low_rot, high_rot, dim, base=10000, max_position_embeddings=2048):
    low = math.floor(find_correction_dim(
        low_rot, dim, base, max_position_embeddings))
    high = math.ceil(find_correction_dim(
        high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim-1)  # Clamp values just in case

def linear_ramp_mask(min, max, dim):
    if min == max:
        max += 0.001  # Prevent singularity

    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func

def _get_mscale_su(scale=1.0, original_max_position_embeddings=None):
    assert original_max_position_embeddings is not None
    seq_len = original_max_position_embeddings
    if scale <= 1.0:
        return 1.0
    # return 0.1 * math.log(scale) + 1.0
    return math.sqrt( math.log(scale*seq_len)/math.log(seq_len) )
    
def _get_mscale_yarn(scale=1.0):
    if scale <= 1.0:
        return 1.0
    return 0.1 * math.log(scale) + 1.0
    
# PI method implementation
def rope_pi(dim, base, scaling_factor, device):
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
    scaling_factor = scaling_factor * 1.0
    # linear interpolation
    inv_freq = inv_freq / scaling_factor
    
    return inv_freq
    
# YaRN method implementation
def rope_yarn(dim, base, original_max_position_embeddings, scaling_factor, device, extrapolation_factor=1, attn_factor=1, beta_fast=32, beta_slow=1, finetuned=False):
    scale = scaling_factor
    # self.yarn(device)
    # get from YaRN github
    pos_freqs = base ** (torch.arange(0, dim, 2).float().to(device) / dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (scale * pos_freqs)

    low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_max_position_embeddings)
    
    inv_freq_mask = (1 - linear_ramp_mask(low, high, dim // 2).float().to(device)) * extrapolation_factor # Get n-d rotational scaling corrected for extrapolation
    # print(inv_freq_mask)
    inv_freq = inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask

    return inv_freq
    # self.register_buffer("inv_freq", inv_freq)
    # self.mscale = float(get_mscale(self.scale) * self.attn_factor) # Get n-d magnitude scaling corrected for interpolation

# s-PI method implementation
def rope_s_pi(dim, base, device, lambda_1):
    base_1 = torch.from_numpy(lambda_1).to(device) # [dim / 2]
    assert base_1.shape[0] == dim // 2 , f"lambda_1 error : {base_1.shape[0]}"
    inv_freq = 1.0 / ( base_1 * ( base ** (torch.arange(0, dim, 2).float().to(device) / dim)) )
    
    return inv_freq

# s-PI Start token method implementation
def rope_s_pi_start(dim, base, device, lambda_1, start_token):
    # TODO: use cos_sin_origin
    base_1 = torch.from_numpy(lambda_1).to(device) # [dim / 2]
    assert base_1.shape[0] == dim // 2 , f"lambda_1 error : {base_1.shape[0]}"

    inv_freq = 1.0 / ( base_1 * ( base ** (torch.arange(0, dim, 2).float().to(device) / dim)) )
    
    return inv_freq


# dynamic s-PI method implementation
def rope_dy_s_pi(dim, base, original_max_position_embeddings, seq_len, scaling_factor, device, lambda_1):
    # scale = scaling_factor
    # NOTE: change scale
    scale = 32.0
    base_1 = torch.from_numpy(lambda_1).to(device) # [dim / 2]
    assert base_1.shape[0] == dim // 2 , f"lambda_1 error : {base_1.shape[0]}"
    # dynamic:$$ 
    base_1 = 1.0 + (base_1 - 1.0) * (seq_len - original_max_position_embeddings)/(original_max_position_embeddings* (scale - 1))
    
    inv_freq = 1.0 / ( base_1 * ( base ** (torch.arange(0, dim, 2).float().to(device) / dim)) )
    
    return inv_freq

@cube.graph.parser.register('bs q_len, 64 -> 1 1 q_len head_dim, 1 1 q_len head_dim')
def apply_rotary_embd(
        position_ids: torch.Tensor,
        lambda_1: torch.Tensor,
        seq_len: int,
        method: str,
        tmps: str,
        scaling_factor: float,
        finetuned: bool,
        start_token: int,
        dim: int,
        base: int,
        original_max_position_embeddings: int, 
        ):
    
    _, q_len = position_ids.shape
    device = torch.device("cuda")
    
    lambda_1 = lambda_1.cpu().numpy()
    
    # TODO: pi yarn spi spi-start dy-spi
    # get inv_freq by 5 methods
    # print(device)
    if method == "pi":
        inv_freq = rope_pi(dim, base, scaling_factor, device)
        
    elif method == "yarn":
        inv_freq = rope_yarn(dim, base, original_max_position_embeddings, scaling_factor, device, finetuned=finetuned)
        
    elif method == "s_pi":
        inv_freq = rope_s_pi(dim, base, device, lambda_1)
        
    elif method == "s_pi_start":
        inv_freq = rope_s_pi_start(dim, base, device, lambda_1, start_token)
        
    elif method == "dy_s_pi":
        inv_freq = rope_dy_s_pi(dim, base, original_max_position_embeddings, seq_len, scaling_factor, device, lambda_1)
        
    # choose tmps
    # mscale_factor = max(seq_len / (1.0*original_max_position_embeddings), 1.0)
    mscale_factor = seq_len / original_max_position_embeddings
    
    if tmps == "su" or method in ["s_pi", "s_pi_start", "dy_s_pi"]:
        mscale = float(_get_mscale_su(mscale_factor, original_max_position_embeddings))
    elif tmps == "yarn" or method == "yarn":
        mscale = float(_get_mscale_yarn(mscale_factor))
    elif tmps == "non" or method == "pi":
        mscale = 1.0
    else:
        raise ValueError(
            f"args.tmps {tmps} must in [su, yarn, non]"
        )
        
    # t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
    t = position_ids.view(-1)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    # [seq_len, dim]
    
    emb_cos = (emb.cos() * mscale)[None, None, :, :]
    emb_sin = (emb.sin() * mscale)[None, None, :, :]
    # [1, 1, seq_len, dim]
    
    if method == "s_pi_start" and start_token > 0:
        AssertionError("Not implemented")
        cos_sin_origin = torch.load("cos_sin_origin.pt")
        assert cos_sin_origin is not None
        emb_origin_cos, emb_origin_sin = cos_sin_origin
        emb_cos[:, :, 0:int(start_token), :] = emb_origin_cos[:, :, 0:int(start_token), :].to(device)
        emb_sin[:, :, 0:int(start_token), :] = emb_origin_sin[:, :, 0:int(start_token), :].to(device)
    
    # Fit for transformers 4.34
    # emb = torch.cat((freqs, freqs), dim=-1)
    # self.register_buffer("cos_cached", emb_cos.to(dtype), persistent=False)
    # self.register_buffer("sin_cached", emb_sin.to(dtype), persistent=False)

    cos = emb_cos[:, :, :q_len, ...].to(dtype=torch.bfloat16).transpose(1, 2)
    sin = emb_sin[:, :, :q_len, ...].to(dtype=torch.bfloat16).transpose(1, 2)
    return cos, sin

class CubeLlamaDynamicScaledRotaryEmbedding(torch.nn.Module):

    def __init__(self, dim, 
                 max_position_embeddings=4096,
                 scale=1.0,
                 base=10000, device=None, 
                 original_max_position_embeddings=4096,
                #  cos_sin_origin=None
                ):
        super().__init__()
        
        self.dim = dim
        self.base = base
        self.max_position_embeddings = max_position_embeddings
        self.original_max_position_embeddings = original_max_position_embeddings
        # self.cos_sin_origin = cos_sin_origin
    # def _set_cos_sin_cache(self, seq_len, device, dtype):
    #     self.max_seq_len_cached = seq_len

    #     dim = self.dim
    #     base = self.base
    #     scaling_factor = max(seq_len / (1.0*self.original_max_position_embeddings), 1.0)
        
    #     base_1 = torch.from_numpy(self.lambda_1).to(device) # [dim / 2]
    #     assert base_1.shape[0] == dim // 2 , f"lambda_1 error : {base_1.shape[0]}"
    #     # print(base_1)
    #     # dynamic:$$ 
    #     base_1 = 1.0 + (base_1 - 1.0) * (seq_len - self.original_max_position_embeddings)/(self.original_max_position_embeddings* (self.scale - 1))
        
                 
    #     if self.tmps:
    #         self.mscale = float(self._get_mscale(scaling_factor))
    #     else:
    #         self.mscale = 1.0       
    #     # self.mscale = float(self._get_mscale(scaling_factor))
    #     # print("self.mscale", self.mscale)
        
    #     inv_freq = 1.0 / ( base_1 * ( self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)) )
        
    #     self.register_buffer("inv_freq", inv_freq)

    #     t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

    #     freqs = torch.einsum("i,j->ij", t, self.inv_freq)
    #     # Different from paper, but it uses a different permutation in order to obtain the same calculation
    #     emb = torch.cat((freqs, freqs), dim=-1)
    #     self.register_buffer("cos_cached", (emb.cos() * self.mscale)[None, None, :, :].to(dtype), persistent=False)
    #     self.register_buffer("sin_cached", (emb.sin() * self.mscale)[None, None, :, :].to(dtype), persistent=False)
        

    def forward(self, 
                lambda_1: torch.Tensor,
                seq_len: int,
                method: str,
                tmps: str,
                scaling_factor: float,
                finetuned: bool,
                start_token: int,
                position_ids: torch.Tensor,
                ):
        
        return apply_rotary_embd(
            position_ids,
            lambda_1,
            seq_len,
            method,
            tmps,
            scaling_factor,
            finetuned,
            start_token,
            self.dim,
            self.base,
            self.original_max_position_embeddings,
        )
        
