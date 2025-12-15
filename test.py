

import os
import re

def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text

def split_text(text, split_ratio=0.9):
    n = int(len(text) * split_ratio)
    return text[:n], text[n:]

if __name__ == "__main__":
    raw1 = load_text("wizofoz.txt")
    raw2 = load_text("romju.txt")
    raw3 = load_text("kjamesbible.txt")
    raw_text = raw1 + "====BOOK2======\n" + raw2 + "======BOOK3======\n" + raw3
    clean = clean_text(raw_text)
    train_text, val_text = split_text(clean)

    print("Total chars:", len(clean))
    print("Train chars:", len(train_text))
    print("Val chars:", len(val_text))

    os.makedirs("data", exist_ok=True)

    with open("data/train.txt", "w", encoding="utf-8") as f:
        f.write(train_text)

    with open("data/val.txt", "w", encoding="utf-8") as f:
        f.write(val_text)
