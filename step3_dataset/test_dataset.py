import sys
from pathlib import Path

# Allow running this file directly: `python step3_dataset/test_dataset.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from step2_tokenizer.tokenizer import CharTokenizer
from step3_dataset.dataset import TextDataset


def main() -> None:
    # load text
    with open("data/cleaned/train.txt", "r", encoding="utf-8") as f:
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


if __name__ == "__main__":
    main()
