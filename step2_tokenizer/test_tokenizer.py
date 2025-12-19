from step2_tokenizer.tokenizer import CharTokenizer


def main() -> None:
    with open("data/cleaned/train.txt", "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = CharTokenizer(text)
    print("Vocab size:", tokenizer.vocab_size)

    sample = text[:200]
    encoded = tokenizer.encode(sample)
    decoded = tokenizer.decode(encoded)

    print("Original:", sample[:100])
    print("Decoded :", decoded[:100])


if __name__ == "__main__":
    main()
