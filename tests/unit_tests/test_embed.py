import torch
import sys
from pathlib import Path

# Allow running this file directly: `python tests/unit_tests/test_embed.py`
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from step2_tokenizer.tokenizer import CharTokenizer
from step3_dataset.dataset import TextDataset
from step4_model.embeddings import GPTEmbeddings

# load data
with open("data/cleaned/train.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = CharTokenizer(text)
tokens = tokenizer.encode(text)

seq_len = 32
dataset = TextDataset(tokens, seq_len)

x, y = dataset[0]
x = x.unsqueeze(0)  # batch size = 1

# model
embed_dim = 64
emb = GPTEmbeddings(
    vocab_size=tokenizer.vocab_size,
    embed_dim=embed_dim,
    seq_len=seq_len
)

out = emb(x)

print("Input shape:", x.shape)
print("Output shape:", out.shape)
