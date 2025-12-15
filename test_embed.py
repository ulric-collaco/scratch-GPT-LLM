import torch
from tokenizer import CharTokenizer
from dataset import TextDataset
from model import GPTEmbeddings

# load data
with open("data/train.txt", "r", encoding="utf-8") as f:
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
