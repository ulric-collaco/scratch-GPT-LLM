import torch
from step4_model.gpt import GPT
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

model = GPT(
    vocab_size=tokenizer.vocab_size,
    seq_len=seq_len,
    embed_dim=64,
    num_heads=4,
    num_layers=4
)

logits = model(x)

print("Input shape :", x.shape)
print("Logits shape:", logits.shape)
