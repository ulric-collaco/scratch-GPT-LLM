import argparse
from pathlib import Path


def merge_text_files(input_files: list[Path]) -> str:
    parts: list[str] = []
    for p in input_files:
        parts.append(p.read_text(encoding="utf-8"))
    return "\n\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 1: merge raw .txt datasets into one file.")
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=str(Path("data") / "raw"),
        help="Folder containing raw .txt files (default: data/raw)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path("data") / "cleaned" / "merged_raw.txt"),
        help="Output merged file (default: data/cleaned/merged_raw.txt)",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Keep the merge deterministic for reproducibility.
    input_files = sorted(raw_dir.glob("*.txt"))
    if not input_files:
        raise FileNotFoundError(f"No .txt files found in {raw_dir}")

    merged = merge_text_files(input_files)
    out_path.write_text(merged, encoding="utf-8")

    print(f"Merged {len(input_files)} files -> {out_path}")


if __name__ == "__main__":
    main()
