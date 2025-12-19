import torch
import torch.nn as nn

from step4_model.embeddings import GPTEmbeddings
from step4_model.transformer_block import TransformerBlock


class GPT(nn.Module):
    def __init__(self, vocab_size: int, seq_len: int, embed_dim: int, num_heads: int, num_layers: int):
        super().__init__()

        self.embed = GPTEmbeddings(vocab_size, embed_dim, seq_len)

        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)]
        )

        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
        self.eval()

        for _ in range(max_new_tokens):
            # keep last seq_len tokens
            idx_cond = idx[:, -self.embed.position_embedding.num_embeddings :]

            logits = self(idx_cond)  # (B, T, vocab)
            logits = logits[:, -1, :]  # last token
            logits = logits / temperature

            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx, next_id], dim=1)

        return idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)

        return logits
