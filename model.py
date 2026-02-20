import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from config import ModelConfig


class RoPE(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        # this is the same as GQA.attn_dim so were using every feature in the head for RoPE
        self.dim = self.config.embedding_dim // self.config.query_heads
        self.cos_cache = None
        self.sin_cache = None

    def build_cache(self, x: torch.Tensor):
        # theta = rope_base ^ (2 * (i - 1) / d)
        # where i is 1..d//2
        # which can be rewritten as rope_base ^ (j / d)
        # where j is 0..d with a step of 2
        j = torch.arange(0, self.dim, 2, device=x.device)[
            : self.dim // 2
        ]  # ensuring we only have d/2 items
        theta = 1 / (self.config.rope_base ** (j.float() / self.dim))
        # possible token positions in curr sequence
        m = torch.arange(
            max(self.config.block_size * 2, x.shape[2]), device=x.device
        )
        # precompute every combination of m and theta
        # shape: [block_size * 2, attn_dim / 2]
        m_theta = torch.outer(m, theta)
        # shape: [block_size * 2, attn_dim]
        self.cos_cache = m_theta.cos().repeat(1, 2).to(x.dtype)
        self.sin_cache = m_theta.sin().repeat(1, 2).to(x.dtype)

    def forward(self, x: torch.Tensor):
        if self.cos_cache is None or self.cos_cache.shape[0] < x.shape[2]:
            self.build_cache(x)

        neg_half_x = torch.cat(
            [-x[:, :, :, self.dim // 2 :], x[:, :, :, : self.dim // 2]], dim=-1
        )

        return (
            x * self.cos_cache[: x.shape[2]].view(1, 1, x.shape[2], x.shape[3])
        ) + (
            neg_half_x
            * self.sin_cache[: x.shape[2]].view(1, 1, x.shape[2], x.shape[3])
        )


class GQA(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        assert self.config.query_heads % self.config.key_value_heads == 0
        self.attn_dim = self.config.embedding_dim // self.config.query_heads
        self.scale_factor = 1 / (self.attn_dim**0.5)

        self.W_q = nn.Linear(
            self.config.embedding_dim,
            self.attn_dim * self.config.query_heads,
            bias=False,
        )
        self.W_k = nn.Linear(
            self.config.embedding_dim,
            self.attn_dim * self.config.key_value_heads,
            bias=False,
        )
        self.W_v = nn.Linear(
            self.config.embedding_dim,
            self.attn_dim * self.config.key_value_heads,
            bias=False,
        )
        self.W_o = nn.Linear(
            self.attn_dim * self.config.query_heads,
            self.config.embedding_dim,
            bias=False,
        )
        self.rope = RoPE(self.config)

    def forward(self, x: torch.Tensor):
        B, L, _ = x.shape

        # shape: B, {self.config.query_heads | self.config.key_value_heads}, L, self.attn_dim
        Q = (
            self.W_q(x)
            .view(B, L, self.config.query_heads, self.attn_dim)
            .transpose(-2, -3)
        )
        K = (
            self.W_k(x)
            .view(B, L, self.config.key_value_heads, self.attn_dim)
            .transpose(-2, -3)
        )
        V = (
            self.W_v(x)
            .view(B, L, self.config.key_value_heads, self.attn_dim)
            .transpose(-2, -3)
        )
        Q = self.rope(Q)
        K = self.rope(K)

        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            attn_weight = F.scaled_dot_product_attention(
                Q,
                K,
                V,
                scale=self.scale_factor,
                is_causal=True,
                enable_gqa=True,
            )

        # attn_weight shape: B, num_heads, L, attn_dims ->  B, L, num_heads, attn_dims
        attn_weight = attn_weight.transpose(-2, -3).contiguous()

        attn_weight = attn_weight.view(B, L, -1)

        return self.W_o(attn_weight)


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.W = nn.Linear(
            self.config.embedding_dim, self.config.hidden_dim, bias=False
        )
        self.V = nn.Linear(
            self.config.embedding_dim, self.config.hidden_dim, bias=False
        )
        self.W2 = nn.Linear(
            self.config.hidden_dim, self.config.embedding_dim, bias=False
        )

    def forward(self, x: torch.Tensor):
        return self.W2(F.silu(self.W(x)) * self.V(x))


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.attn_norm = nn.RMSNorm(
            self.config.embedding_dim, self.config.norm_eps
        )
        self.attn = GQA(config)

        self.ffn_norm = nn.RMSNorm(
            self.config.embedding_dim, self.config.norm_eps
        )
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.attn_norm(x))
        return x + self.ffn(self.ffn_norm(x))


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(
            self.config.vocab_size, self.config.embedding_dim
        )
        self.layers = nn.ModuleList()
        for _ in range(self.config.layers):
            self.layers.append(Block(self.config))
        self.norm = nn.RMSNorm(self.config.embedding_dim, self.config.norm_eps)
        self.lm_head = nn.Linear(
            self.config.embedding_dim, self.config.vocab_size
        )

    def forward(self, tokens):
        x = self.embedding(tokens)
        for layer in self.layers:
            x = layer(x)
        x = self.lm_head(self.norm(x))
        return x
