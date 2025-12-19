import argparse
from pathlib import Path


def split_text(text: str, split_ratio: float = 0.9) -> tuple[str, str]:
    n = int(len(text) * split_ratio)
    return text[:n], text[n:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 1: split cleaned text into train/val.")
    parser.add_argument(
        "--in",
        dest="in_path",
        type=str,
        default=str(Path("data") / "cleaned" / "merged_clean.txt"),
        help="Input cleaned file (default: data/cleaned/merged_clean.txt)",
    )
    parser.add_argument(
        "--train-out",
        type=str,
        default=str(Path("data") / "cleaned" / "train.txt"),
        help="Train output path (default: data/cleaned/train.txt)",
    )
    parser.add_argument(
        "--val-out",
        type=str,
        default=str(Path("data") / "cleaned" / "val.txt"),
        help="Val output path (default: data/cleaned/val.txt)",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.9,
        help="Train split ratio (default: 0.9)",
    )
    args = parser.parse_args()

    in_path = Path(args.in_path)
    train_out = Path(args.train_out)
    val_out = Path(args.val_out)

    train_out.parent.mkdir(parents=True, exist_ok=True)
    val_out.parent.mkdir(parents=True, exist_ok=True)

    text = in_path.read_text(encoding="utf-8")
    train_text, val_text = split_text(text, split_ratio=args.ratio)

    train_out.write_text(train_text, encoding="utf-8")
    val_out.write_text(val_text, encoding="utf-8")

    print(f"Total chars: {len(text)}")
    print(f"Train chars: {len(train_text)} -> {train_out}")
    print(f"Val chars: {len(val_text)} -> {val_out}")


if __name__ == "__main__":
    main()
