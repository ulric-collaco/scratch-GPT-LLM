from tokenizer import CharTokenizer
from dataset import TextDataset

# load text
with open("data/train.txt", "r", encoding="utf-8") as f:
    text = f.read()

# tokenize
tokenizer = CharTokenizer(text)
tokens = tokenizer.encode(text)

print("Total tokens:", len(tokens))

# dataset
seq_len = 32
dataset = TextDataset(tokens, seq_len)

# inspect one sample
x, y = dataset[0]

print("X:", x)
print("Y:", y)
print("Decoded X:", tokenizer.decode(x.tolist()))
print("Decoded Y:", tokenizer.decode(y.tolist()))
