import argparse
import re
from pathlib import Path


def clean_text(text: str) -> str:
    # Minimal cleaning: normalize whitespace.
    # (Keeps the project educational and reproducible.)
    return re.sub(r"\s+", " ", text).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 1: clean merged text (whitespace normalization).")
    parser.add_argument(
        "--in",
        dest="in_path",
        type=str,
        default=str(Path("data") / "cleaned" / "merged_raw.txt"),
        help="Input text file (default: data/cleaned/merged_raw.txt)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path("data") / "cleaned" / "merged_clean.txt"),
        help="Output cleaned file (default: data/cleaned/merged_clean.txt)",
    )
    args = parser.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    text = in_path.read_text(encoding="utf-8")
    out_path.write_text(clean_text(text), encoding="utf-8")

    print(f"Cleaned -> {out_path}")


if __name__ == "__main__":
    main()
