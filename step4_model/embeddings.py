import torch
import torch.nn as nn


class GPTEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, seq_len: int):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(seq_len, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len)
        batch_size, seq_len = x.shape

        # positions: [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=x.device)
        positions = positions.unsqueeze(0)  # shape: (1, seq_len)

        token_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(positions)

        return token_emb + pos_emb
