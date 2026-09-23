"""
El GPT 1.0 Neural Network Architecture
Modern Decoder-only Transformer with:
- Rotary Position Embeddings (RoPE)
- Root Mean Square Normalization (RMSNorm)
- SwiGLU Gated Feed-Forward Layers
- PyTorch Scaled Dot-Product Causal Attention (MPS / Apple Silicon optimized)
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ElGPTConfig:
    vocab_size: int = 8192
    d_model: int = 288
    n_layers: int = 6
    n_heads: int = 6
    max_seq_len: int = 512
    dropout: float = 0.0
    multiple_of: int = 64
    norm_eps: float = 1e-5

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @classmethod
    def preset_10m(cls, vocab_size: int = 1046) -> "ElGPTConfig":
        """Fast training configuration for ~10M parameter model."""
        return cls(
            vocab_size=vocab_size,
            d_model=288,
            n_layers=6,
            n_heads=6,
            max_seq_len=512,
        )

    @classmethod
    def preset_100m(cls, vocab_size: int = 16384) -> "ElGPTConfig":
        """Configuration for ~100M parameter model (768 dim, 12 layers, 12 heads)."""
        return cls(
            vocab_size=vocab_size,
            d_model=768,
            n_layers=12,
            n_heads=12,
            max_seq_len=512,
        )

    @classmethod
    def preset_500m(cls, vocab_size: int = 32768) -> "ElGPTConfig":
        """Configuration for ~490M parameter model (1024 dim, 24 layers, 16 heads)."""
        return cls(
            vocab_size=vocab_size,
            d_model=1024,
            n_layers=24,
            n_heads=16,
            max_seq_len=512,
        )

    @classmethod
    def preset_1b(cls, vocab_size: int = 32768) -> "ElGPTConfig":
        """Configuration for ~1 Billion parameter model (1536 dim, 32 layers, 16 heads)."""
        return cls(
            vocab_size=vocab_size,
            d_model=1536,
            n_layers=32,
            n_heads=16,
            max_seq_len=512,
        )


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(x.float()).type_as(x) * self.weight


def precompute_rope_frequencies(dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    # Return complex exponential cis(theta) = cos + i*sin
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # xq: (batch, seq_len, n_heads, head_dim)
    # freqs_cis: (seq_len, head_dim // 2)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, head_dim // 2)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ElGPTConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.d_model = config.d_model

        self.wq = nn.Linear(config.d_model, config.d_model, bias=False)
        self.wk = nn.Linear(config.d_model, config.d_model, bias=False)
        self.wv = nn.Linear(config.d_model, config.d_model, bias=False)
        self.wo = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, S, _ = x.shape

        q = self.wq(x).view(B, S, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, S, self.n_heads, self.head_dim)
        v = self.wv(x).view(B, S, self.n_heads, self.head_dim)

        q, k = apply_rotary_emb(q, k, freqs_cis)

        if kv_cache is not None:
            prev_k, prev_v = kv_cache
            k = torch.cat([prev_k, k], dim=1)
            v = torch.cat([prev_v, v], dim=1)
        new_kv_cache = (k, v)

        # Transpose for scaled dot product attention: (B, n_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Use PyTorch built-in SDPA (leverages Metal / Flash kernels where supported)
        is_causal = mask is None and S > 1 and kv_cache is None
        output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.config.dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        output = output.transpose(1, 2).contiguous().view(B, S, self.d_model)
        return self.wo(output), new_kv_cache


class SwiGLU(nn.Module):
    def __init__(self, config: ElGPTConfig):
        super().__init__()
        hidden_dim = int(2 * (4 * config.d_model) / 3)
        hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)

        self.w1 = nn.Linear(config.d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.d_model, bias=False)
        self.w3 = nn.Linear(config.d_model, hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class ElGPTBlock(nn.Module):
    def __init__(self, config: ElGPTConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.ffn = SwiGLU(config)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        h, new_cache = self.attn(self.attn_norm(x), freqs_cis, mask=mask, kv_cache=kv_cache)
        x = x + h
        x = x + self.ffn(self.ffn_norm(x))
        return x, new_cache


class ElGPTModel(nn.Module):
    def __init__(self, config: ElGPTConfig):
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([ElGPTBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Tie weights between token embeddings and LM head for parameter efficiency
        self.lm_head.weight = self.tok_embeddings.weight

        # Precompute RoPE complex frequencies
        freqs_cis = precompute_rope_frequencies(config.head_dim, config.max_seq_len)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_caches: Optional[list] = None,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[list]]:
        B, S = input_ids.shape
        h = self.tok_embeddings(input_ids)

        freqs_cis = self.freqs_cis[start_pos : start_pos + S]

        new_caches = []
        for i, layer in enumerate(self.layers):
            cache_i = kv_caches[i] if kv_caches is not None else None
            h, new_cache = layer(h, freqs_cis, kv_cache=cache_i)
            if kv_caches is not None:
                new_caches.append(new_cache)

        h = self.norm(h)
        logits = self.lm_head(h)

        loss = None
        if targets is not None:
            # Flatten across sequence dimension for cross entropy
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=-100,
            )

        return logits, loss, new_caches if kv_caches is not None else None


if __name__ == "__main__":
    cfg = ElGPTConfig()
    model = ElGPTModel(cfg)
    num_params = model.get_num_params()
    print(f"[El GPT 1.0 Architecture] Initialized with {num_params:,} parameters ({num_params / 1e6:.2f}M).")
    dummy_input = torch.randint(0, cfg.vocab_size, (2, 32))
    dummy_target = torch.randint(0, cfg.vocab_size, (2, 32))
    logits, loss, _ = model(dummy_input, targets=dummy_target)
    print(f"Forward pass successful. Logits shape: {logits.shape}, Loss: {loss.item():.4f}")
