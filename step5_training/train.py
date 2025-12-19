import argparse
import csv
import json
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Allow running this file directly: `python step5_training/train.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from step2_tokenizer.tokenizer import CharTokenizer
from step3_dataset.dataset import TextDataset
from step3_dataset.stream_dataset import StreamingTokenBatchLoader
from step1_data_prep.hf_loader import scan_hf_for_charset_and_count, stream_token_ids_from_hf
from step1_data_prep.parquet_stream import (
    scan_parquet_for_charset_and_count,
    stream_token_ids_from_parquet,
)
from step4_model.gpt import GPT


def _ensure_dir(path: Path) -> None:
    """MLOps: ensure output directories exist."""
    path.mkdir(parents=True, exist_ok=True)


def _latest_checkpoint(checkpoints_dir: Path) -> Tuple[Optional[Path], int]:
    """MLOps: find the latest checkpoint and its epoch number.

    Expected filenames: epoch_{epoch}.pt
    Returns (path, epoch). If none found, returns (None, 0).
    """
    if not checkpoints_dir.exists():
        return None, 0

    best_epoch = 0
    best_path: Optional[Path] = None

    for p in checkpoints_dir.glob("epoch_*.pt"):
        m = re.match(r"^epoch_(\d+)\.pt$", p.name)
        if not m:
            continue
        epoch = int(m.group(1))
        if epoch > best_epoch:
            best_epoch = epoch
            best_path = p

    return best_path, best_epoch


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a from-scratch character-level GPT.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the latest checkpoint in checkpoints/.",
    )
    args = parser.parse_args()

    # ------------------
    # config (keep the same hyperparameters)
    # ------------------
    seq_len = 32
    batch_size = 32
    embed_dim = 64
    num_heads = 4
    num_layers = 2
    lr = 3e-4
    epochs = 15

    # Keep script runnable on CPU as well.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------
    # MLOps: project-relative paths
    # ------------------
    # This makes the script runnable from any working directory.
    project_root = Path(__file__).resolve().parents[1]

    checkpoints_dir = project_root / "checkpoints"
    _ensure_dir(checkpoints_dir)

    artifacts_dir = project_root / "step5_training"
    _ensure_dir(artifacts_dir)

    config_path = artifacts_dir / "config.json"
    log_path = artifacts_dir / "training_log.csv"
    tokenizer_path = artifacts_dir / "tokenizer.pkl"
    final_model_path = artifacts_dir / "gpt_char_model.pt"

    # ------------------
    # MLOps: persist training configuration (at training start)
    # ------------------
    config = {
        "seq_len": seq_len,
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "lr": lr,
        "batch_size": batch_size,
        "epochs": epochs,
    }
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # ------------------
    # data
    # ------------------
    # Default mode: Hugging Face streaming (automatic download).
    # Manual mode (opt-in): if you drop any *.parquet files into data/raw/, use local Parquet streaming.
    parquet_dir = project_root / "data" / "raw"
    local_parquets = sorted(parquet_dir.glob("*.parquet"))

    if local_parquets:
        # --- Manual mode: Local Parquet streaming ---
        # Text column defaults to "text" for common HF parquet exports.
        text_column = "text"
        parquet_batch_size = 1024

        scan = scan_parquet_for_charset_and_count(
            local_parquets,
            text_column=text_column,
            batch_size=parquet_batch_size,
        )
        tokenizer = CharTokenizer("".join(sorted(scan.chars)))

        with tokenizer_path.open("wb") as f:
            pickle.dump(tokenizer, f)

        def _token_iter_factory():
            return stream_token_ids_from_parquet(
                local_parquets,
                tokenizer=tokenizer,
                text_column=text_column,
                parquet_batch_size=parquet_batch_size,
            )

        loader = StreamingTokenBatchLoader(
            token_iter_factory=_token_iter_factory,
            seq_len=seq_len,
            batch_size=batch_size,
            total_tokens=scan.total_chars,
        )
    else:
        # --- Default mode: Hugging Face streaming ---
        hf_dataset_name = "wikitext"
        hf_dataset_config = "wikitext-103-raw-v1"
        hf_split = "train"
        hf_text_column = "text"

        scan = scan_hf_for_charset_and_count(
            dataset_name=hf_dataset_name,
            dataset_config=hf_dataset_config,
            split=hf_split,
            text_column=hf_text_column,
        )
        tokenizer = CharTokenizer("".join(sorted(scan.chars)))

        with tokenizer_path.open("wb") as f:
            pickle.dump(tokenizer, f)

        def _token_iter_factory():
            return stream_token_ids_from_hf(
                tokenizer=tokenizer,
                dataset_name=hf_dataset_name,
                dataset_config=hf_dataset_config,
                split=hf_split,
                text_column=hf_text_column,
            )

        loader = StreamingTokenBatchLoader(
            token_iter_factory=_token_iter_factory,
            seq_len=seq_len,
            batch_size=batch_size,
            total_tokens=scan.total_chars,
        )

    # ------------------
    # model (keep architecture unchanged)
    # ------------------
    model = GPT(
        vocab_size=tokenizer.vocab_size,
        seq_len=seq_len,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # ------------------
    # MLOps: resume training from latest checkpoint
    # ------------------
    start_epoch = 0  # completed epochs; training loop below uses 1..epochs
    if args.resume:
        ckpt_path, ckpt_epoch = _latest_checkpoint(checkpoints_dir)
        if ckpt_path is None:
            print("--resume provided but no checkpoints found; starting from scratch.")
        else:
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            start_epoch = ckpt_epoch
            print(f"Resumed from {ckpt_path} (epoch {ckpt_epoch}).")

    # ------------------
    # MLOps: loss logging setup
    # ------------------
    # Requirement: Save loss per epoch to training_log.csv with columns epoch, loss
    log_needs_header = not log_path.exists()
    log_file = log_path.open("a", newline="", encoding="utf-8")
    log_writer = csv.DictWriter(log_file, fieldnames=["epoch", "loss"])
    if log_needs_header:
        log_writer.writeheader()

    # ------------------
    # training loop (keep logic unchanged)
    # ------------------
    model.train()
    start_time = time.time()

    for epoch in range(start_epoch + 1, epochs + 1):
        epoch_start = time.time()
        total_loss = 0.0

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)  # (B, T, vocab)
            B, T, V = logits.shape

            loss = criterion(
                logits.view(B * T, V),
                y.view(B * T),
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())

        avg_loss = total_loss / len(loader)
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Time: {epoch_time / 60:.2f} min")

        # ------------------
        # MLOps: checkpointing (end of every epoch)
        # ------------------
        # Requirement: Save model.state_dict() at end of every epoch
        # Requirement: checkpoints/epoch_{epoch}.pt
        ckpt_out = checkpoints_dir / f"epoch_{epoch}.pt"
        torch.save(model.state_dict(), ckpt_out)

        # MLOps: write training loss per epoch
        log_writer.writerow({"epoch": epoch, "loss": avg_loss})
        log_file.flush()

    total_time = time.time() - start_time
    print(f"Total training time: {total_time / 60:.2f} min")

    # Keep a final artifact for convenient inference.
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved model to {final_model_path}")

    log_file.close()


if __name__ == "__main__":
    main()
