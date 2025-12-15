import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tokenizer import CharTokenizer
from dataset import TextDataset
from model import GPT

# ------------------
# config
# ------------------
seq_len = 32
batch_size = 32
embed_dim = 64
num_heads = 4
num_layers = 2
lr = 3e-4
epochs = 5
device = "cuda" 

# ------------------
# data
# ------------------
with open("data/train.txt", "r", encoding="utf-8") as f:
    train_text = f.read()

tokenizer = CharTokenizer(train_text)
train_tokens = tokenizer.encode(train_text)

dataset = TextDataset(train_tokens, seq_len)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ------------------
# model
# ------------------
model = GPT(
    vocab_size=tokenizer.vocab_size,
    seq_len=seq_len,
    embed_dim=embed_dim,
    num_heads=num_heads,
    num_layers=num_layers
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# ------------------
# training loop
# ------------------
model.train()

start_time = time.time()

for epoch in range(epochs):
    epoch_start = time.time()
    total_loss = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)  # (B, T, vocab)
        B, T, V = logits.shape

        loss = criterion(
            logits.view(B * T, V),
            y.view(B * T)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    epoch_time = time.time() - epoch_start
    print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s")

total_time = time.time() - start_time
print(f"Total training time: {total_time:.2f}s")

torch.save(model.state_dict(), "gpt_char_model.pt")
print("Saved model to gpt_char_model.pt")
