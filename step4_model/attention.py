import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()

        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        B, T, C = x.shape

        Q = self.q(x)  # (B, T, C)
        K = self.k(x)  # (B, T, C)
        V = self.v(x)  # (B, T, C)

        # attention scores
        scores = Q @ K.transpose(-2, -1) / math.sqrt(C)
        # scores: (B, T, T)

        # causal mask (no looking right)
        mask = torch.tril(torch.ones(T, T, device=x.device))
        scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        out = attn @ V  # (B, T, C)

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.qkv(x)  # (B, T, 3C)
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        Q, K, V = qkv[0], qkv[1], qkv[2]
        # each: (B, heads, T, head_dim)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        mask = torch.tril(torch.ones(T, T, device=x.device))
        scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        out = attn @ V  # (B, heads, T, head_dim)

        out = out.transpose(1, 2).reshape(B, T, C)
        return self.out(out)
