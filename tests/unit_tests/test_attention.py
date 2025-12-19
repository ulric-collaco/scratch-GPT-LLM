import torch
from step4_model.embeddings import GPTEmbeddings
from step4_model.attention import SelfAttention
from step2_tokenizer.tokenizer import CharTokenizer
from step3_dataset.dataset import TextDataset

with open("data/cleaned/train.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = CharTokenizer(text)
tokens = tokenizer.encode(text)

seq_len = 32
dataset = TextDataset(tokens, seq_len)
x, _ = dataset[0]
x = x.unsqueeze(0)

embed_dim = 64

emb = GPTEmbeddings(tokenizer.vocab_size, embed_dim, seq_len)
attn = SelfAttention(embed_dim)

h = emb(x)
out = attn(h)

print("Embedding output:", h.shape)
print("Attention output:", out.shape)
