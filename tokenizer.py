class CharTokenizer:
    def __init__(self, text):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text):
        return [self.stoi[ch] for ch in text]

    def decode(self, ids):
        return ''.join([self.itos[i] for i in ids])

if __name__ == "__main__":
    with open("data/train.txt", "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = CharTokenizer(text)

    print("Vocab size:", tokenizer.vocab_size)

    sample = text[:200]
    encoded = tokenizer.encode(sample)
    decoded = tokenizer.decode(encoded)

    print("Original:", sample[:100])
    print("Decoded :", decoded[:100])