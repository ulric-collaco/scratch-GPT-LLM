import torch
from model import GPT
from tokenizer import CharTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

# load text (only for tokenizer)
with open("data/train.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = CharTokenizer(text)

# build model
model = GPT(
    vocab_size=tokenizer.vocab_size,
    seq_len=32,
    embed_dim=64,
    num_heads=4,
    num_layers=2
).to(device)

# load trained weights
model.load_state_dict(torch.load("gpt_char_model.pt", map_location=device))

# starting prompt
prompt = "the project gutenberg"
idx = torch.tensor([tokenizer.encode(prompt)], device=device)

# generate
out = model.generate(idx, max_new_tokens=300, temperature=0.8)

print(tokenizer.decode(out[0].tolist()))
