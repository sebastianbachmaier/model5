"""
model.py — Llama-style decoder-only transformer, implemented from scratch.

Architecture (defaults produce ~0.5B params):
  dim=1280, n_layers=26, n_heads=20, n_kv_heads=5 (GQA), max_seq_len=2048,
  vocab_size=32768, RMSNorm, RoPE, SwiGLU FFN, weight-tied embeddings.

No HuggingFace model classes are used here — only plain nn.Module / functional
PyTorch, so every piece of the forward pass is explicit and hackable.
"""
import math
from dataclasses import dataclass, asdict
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    dim: int = 1280
    n_layers: int = 26
    n_heads: int = 20
    n_kv_heads: int = 5
    vocab_size: int = 32768
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    multiple_of: int = 256  # SwiGLU hidden dim is rounded up to a multiple of this

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def swiglu_hidden_dim(dim: int, multiple_of: int) -> int:
    """Standard Llama SwiGLU sizing: two up-projections + one down-projection
    cost as much compute as a single 4*dim MLP would with *three* matrices, so
    the hidden dim is shrunk by 2/3 first, then rounded up to `multiple_of`
    (256) so matmul shapes are hardware-friendly."""
    hidden = 4 * dim
    hidden = int(2 * hidden / 3)
    hidden = multiple_of * ((hidden + multiple_of - 1) // multiple_of)
    return hidden


class RMSNorm(nn.Module):
    """RMSNorm computed in fp32 for numerical stability even under bf16
    autocast, then cast back to the input dtype. No bias/mean-centering
    (that's the whole point of RMSNorm vs LayerNorm)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(in_dtype)


def precompute_rope(head_dim: int, max_seq_len: int, theta: float, device=None):
    """Precompute cos/sin tables for RoPE, shape (max_seq_len, head_dim/2).
    Uses the standard "rotate half" (non-interleaved) convention: the head
    dim is split into two halves and rotated as a pair, which is equivalent
    to the interleaved complex-number formulation but is easier to make fast
    on GPUs and matches most modern Llama-style implementations."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)  # (max_seq_len, head_dim/2)
    return torch.cos(freqs), torch.sin(freqs)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply RoPE to queries and keys.
    xq, xk: (bsz, seqlen, n_heads, head_dim); cos, sin: (seqlen, head_dim/2).
    We duplicate cos/sin across the two halves of head_dim so the same
    rotation angle is used for (x_i, x_{i+d/2}) pairs, per the RoPE paper."""
    cos = torch.cat([cos, cos], dim=-1)[None, :, None, :]  # (1, seqlen, 1, head_dim)
    sin = torch.cat([sin, sin], dim=-1)[None, :, None, :]
    xq_out = xq * cos + rotate_half(xq) * sin
    xk_out = xk * cos + rotate_half(xk) * sin
    return xq_out.to(xq.dtype), xk_out.to(xk.dtype)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Grouped-query attention: expand n_kv_heads -> n_heads by repeating each
    KV head n_rep times so it lines up with the corresponding group of query
    heads. This is done right before attention so that we can still *cache*
    only the un-repeated (small) KV tensors during generation."""
    if n_rep == 1:
        return x
    bsz, seqlen, n_kv_heads, head_dim = x.shape
    return (
        x[:, :, :, None, :]
        .expand(bsz, seqlen, n_kv_heads, n_rep, head_dim)
        .reshape(bsz, seqlen, n_kv_heads * n_rep, head_dim)
    )


class KVCache:
    """Holds the un-repeated (n_kv_heads, not n_heads) K/V tensors for one
    layer during autoregressive decoding. Exploiting GQA here means the cache
    is n_heads / n_kv_heads (=4x for our config) smaller than a standard MHA
    cache would be."""

    def __init__(self, batch_size, max_seq_len, n_kv_heads, head_dim, device, dtype):
        shape = (batch_size, max_seq_len, n_kv_heads, head_dim)
        self.cache_k = torch.zeros(shape, device=device, dtype=dtype)
        self.cache_v = torch.zeros(shape, device=device, dtype=dtype)

    def update(self, start_pos: int, xk: torch.Tensor, xv: torch.Tensor):
        seqlen = xk.size(1)
        self.cache_k[:, start_pos:start_pos + seqlen] = xk
        self.cache_v[:, start_pos:start_pos + seqlen] = xv
        return self.cache_k[:, :start_pos + seqlen], self.cache_v[:, :start_pos + seqlen]


class Attention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_rep = config.n_heads // config.n_kv_heads
        self.head_dim = config.head_dim

        self.wq = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)

    def forward(self, x, cos, sin, kv_cache: Optional[KVCache] = None, start_pos: int = 0):
        bsz, seqlen, _ = x.shape
        xq = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = self.wk(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, cos, sin)

        if kv_cache is not None:
            # Write the new K/V into the cache and read back everything seen
            # so far (positions [0, start_pos+seqlen)).
            xk, xv = kv_cache.update(start_pos, xk, xv)

        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        xq = xq.transpose(1, 2)  # (bsz, n_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)  # (bsz, n_heads, kv_seqlen, head_dim)
        xv = xv.transpose(1, 2)

        # is_causal=True is correct in all three cases we use it:
        #  - full-sequence training (q_len == k_len): standard causal mask.
        #  - cache prefill (q_len == k_len == prompt_len): standard causal mask.
        #  - cache decode (q_len == 1, k_len == start_pos+1): SDPA aligns the
        #    causal mask to the bottom-right corner, so the single query
        #    position is allowed to attend to all cached + current keys.
        out = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(out)


class FeedForward(nn.Module):
    """SwiGLU: two up-projections (w1 gate, w3 value) combined with a SiLU
    gate, then projected back down with w2."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        hidden_dim = swiglu_hidden_dim(config.dim, config.multiple_of)
        self.w1 = nn.Linear(config.dim, hidden_dim, bias=False)  # gate
        self.w3 = nn.Linear(config.dim, hidden_dim, bias=False)  # up
        self.w2 = nn.Linear(hidden_dim, config.dim, bias=False)  # down

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention = Attention(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.feed_forward = FeedForward(config)

    def forward(self, x, cos, sin, kv_cache=None, start_pos=0):
        # Pre-norm residual blocks (norm -> sublayer -> add), as in Llama.
        h = x + self.attention(self.attention_norm(x), cos, sin, kv_cache, start_pos)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layers))
        self.norm = RMSNorm(config.dim, config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        # Weight tying: the same matrix is used to embed tokens and to
        # project the final hidden state back to vocab logits.
        self.output.weight = self.tok_embeddings.weight

        cos, sin = precompute_rope(config.head_dim, config.max_seq_len, config.rope_theta)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

        self.apply(self._init_weights)
        self._scaled_output_init()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _scaled_output_init(self):
        """Re-init the two "output" projections of each sub-layer (attention
        wo, FFN w2) with a smaller std that shrinks with depth. This keeps
        the variance of the residual stream from growing with n_layers,
        which is the standard GPT-2 / Llama initialization trick."""
        std = 0.02 / math.sqrt(2 * self.config.n_layers)
        for name, p in self.named_parameters():
            if name.endswith("wo.weight") or name.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=std)

    def resize_token_embeddings(self, new_vocab_size: int):
        """Grow the embedding / (tied) output matrix in-place, e.g. when SFT
        adds new special role tokens on top of a pretrained checkpoint. New
        rows are initialized the same way as the original embedding init."""
        old_vocab_size, dim = self.tok_embeddings.weight.shape
        if new_vocab_size == old_vocab_size:
            return
        assert new_vocab_size > old_vocab_size, "can only grow embeddings, not shrink"
        new_embed = nn.Embedding(new_vocab_size, dim).to(self.tok_embeddings.weight.device,
                                                           dtype=self.tok_embeddings.weight.dtype)
        nn.init.normal_(new_embed.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            new_embed.weight[:old_vocab_size] = self.tok_embeddings.weight
        self.tok_embeddings = new_embed
        new_output = nn.Linear(dim, new_vocab_size, bias=False).to(new_embed.weight.device,
                                                                     dtype=new_embed.weight.dtype)
        self.output = new_output
        self.output.weight = self.tok_embeddings.weight  # re-tie
        self.config.vocab_size = new_vocab_size

    def forward(self, tokens, targets=None, kv_caches: Optional[List[KVCache]] = None, start_pos: int = 0):
        bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)

        cos = self.cos_cached[start_pos:start_pos + seqlen].to(h.device)
        sin = self.sin_cached[start_pos:start_pos + seqlen].to(h.device)

        for i, layer in enumerate(self.layers):
            cache = kv_caches[i] if kv_caches is not None else None
            h = layer(h, cos, sin, kv_cache=cache, start_pos=start_pos)

        h = self.norm(h)
        logits = self.output(h)

        loss = None
        if targets is not None:
            # targets is already the "next token" for each input position
            # (the shift happens once, in the data pipeline) so no further
            # shifting is needed here. -100 marks positions to ignore, used
            # both for padding and for SFT's non-assistant tokens.
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100
            )
        return logits, loss


def count_parameters(model: nn.Module) -> int:
    """Total trainable parameter count. Tied weights (tok_embeddings /
    output) are automatically de-duplicated by nn.Module.parameters()."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick sanity check: build the default ~0.5B config and print param count.
    cfg = ModelConfig()
    m = Transformer(cfg)
    n = count_parameters(m)
    print(f"config: {cfg}")
    print(f"ffn hidden dim: {swiglu_hidden_dim(cfg.dim, cfg.multiple_of)}")
    print(f"total parameters: {n:,} ({n / 1e9:.3f}B)")
