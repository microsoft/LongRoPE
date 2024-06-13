import math
import torch
import transformers


_UNSQUEEZE_CACHE = int(transformers.__version__.split('.')[1]) < 36


class LlamaLongRoPEScaledRotaryEmbedding(torch.nn.Module):
    """
    LongRoPE Scaled Rotary Positional Encoding class for Llama-like model.

    Args:
        dim (int): Head dimension.
        rescale_factors (list): List of rescale factors for each dimension.
        scale (float, optional): Length scale for code compatibility.
        max_position_embeddings (int, optional): Maximum number of position embeddings (after scaled).
        original_max_position_embeddings (int, optional): Original maximum number of position embeddings (before scaled).
        base (int, optional): Base value for the positional encoding. Defaults to 10000.
        magnitude_scaling_policy (str, optional): Attention temperature scaling function. Defaults to "su".
        device (torch.device, optional): Device on which to create the embedding. Defaults to None.
    """

    def __init__(
        self,
        dim, 
        rescale_factors,
        scale=1.0,
        max_position_embeddings=4096,
        original_max_position_embeddings=4096,
        base=10000,
        magnitude_scaling_policy="su",
        device=None,
    ):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.original_max_position_embeddings = original_max_position_embeddings
        self.base = base

        if magnitude_scaling_policy == "su":
            calc_mscale = self._calc_mscale_su
        elif magnitude_scaling_policy == "yarn":
            calc_mscale = self._calc_mscale_yarn
        else:
            calc_mscale = lambda scale: float(magnitude_scaling_policy)
        self.mscale = calc_mscale(self.max_position_embeddings / self.original_max_position_embeddings)

        rescale_factors = torch.tensor(rescale_factors, dtype=torch.float32, device=device)
        assert rescale_factors.shape == (self.dim // 2, ), \
            f"misaligned shape for LongRoPE rescale factors: {rescale_factors.shape}"

        inv_freq = 1.0 / (rescale_factors * (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _calc_mscale_su(self, scale):
        if scale <= 1.0:
            return 1.0
        return math.sqrt(1 + math.log(scale) / math.log(self.original_max_position_embeddings))

    def _calc_mscale_yarn(self, scale):
        if scale <= 1.0:
            return 1.0
        return 0.1 * math.log(scale) + 1.0

    def forward(self, x, seq_len=None):
        t = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        inv_freq = self.inv_freq.to(x.device)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return (emb.cos() * self.mscale).to(x.dtype), (emb.sin() * self.mscale).to(x.dtype)


class LlamaDynamicLongRoPEScaledRotaryEmbedding(LlamaLongRoPEScaledRotaryEmbedding):

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        dynamic_scale = ((seq_len / self.original_max_position_embeddings) - 1) / (self.scale - 1)
        rescale_factors = 1.0 + (self.rescale_factors - 1.0) * dynamic_scale
        inv_freq = 1.0 / (rescale_factors * (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)))
        self.register_buffer("inv_freq", inv_freq)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)

        mscale = self._calc_mscale(seq_len / self.original_max_position_embeddings)

        emb = torch.cat((freqs, freqs), dim=-1)
        if _UNSQUEEZE_CACHE:
            emb_cos = (emb.cos() * mscale)[None, None, :, :]
            emb_sin = (emb.sin() * mscale)[None, None, :, :]
        else:
            emb_cos = emb.cos() * mscale
            emb_sin = emb.sin() * mscale

        self.register_buffer("cos_cached", emb_cos.to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb_sin.to(dtype), persistent=False)


class LlamaMixedLongRoPEScaledRotaryEmbedding(LlamaLongRoPEScaledRotaryEmbedding):

    def __init__(
        self,
        dim, 
        rescale_factors,
        max_position_embeddings=4096,
        original_max_position_embeddings=4096,
        base=10000,
        magnitude_scaling_policy="su",
        start_token_idx=0,
        original_embeddings=None,
        device=None,
    ):
        self.start_token_idx = start_token_idx
        self.original_embeddings = original_embeddings
        
        super().__init__(
            dim=dim,
            rescale_factors=rescale_factors,
            max_position_embeddings=max_position_embeddings,
            original_max_position_embeddings=original_max_position_embeddings,
            base=base,
            magnitude_scaling_policy=magnitude_scaling_policy,
            device=device,
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        inv_freq = 1.0 / (self.rescale_factors * (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)))
        self.register_buffer("inv_freq", inv_freq)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)

        mscale = self._calc_mscale(seq_len / self.original_max_position_embeddings)

        emb = torch.cat((freqs, freqs), dim=-1)
        if _UNSQUEEZE_CACHE:
            emb_cos = (emb.cos() * mscale)[None, None, :, :]
            emb_sin = (emb.sin() * mscale)[None, None, :, :]
        else:
            emb_cos = emb.cos() * mscale
            emb_sin = emb.sin() * mscale

        if self.start_token_idx > 0:
            assert self.original_embeddings is not None, \
                'need input original embeddings for start token index > 0'
            emb_cos_origin, emb_sin_origin = self.original_embeddings
            assert emb_cos_origin.shape == emb_cos.shape and emb_sin_origin.shape == emb_cos.shape, \
                'original embeddings shape should be the same with current embeddings'
            if _UNSQUEEZE_CACHE:
                emb_cos[:, :, 0:self.start_token_idx, :] = emb_cos_origin[:, :, 0:self.start_token_idx, :]
                emb_sin[:, :, 0:self.start_token_idx, :] = emb_sin_origin[:, :, 0:self.start_token_idx, :]
            else:
                emb_cos[0:self.start_token_idx, :] = emb_cos_origin[0:self.start_token_idx, :]
                emb_sin[0:self.start_token_idx, :] = emb_sin_origin[0:self.start_token_idx, :]

        self.register_buffer("cos_cached", emb_cos.to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb_sin.to(dtype), persistent=False)
