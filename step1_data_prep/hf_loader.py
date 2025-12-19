"""Hugging Face streaming loader (default data mode).

This module provides a minimal, educational interface:
- Stream text rows from a HF dataset using `datasets.load_dataset(..., streaming=True)`.
- Scan once (streaming) to collect charset + count characters.
- Stream token IDs on-the-fly using the existing CharTokenizer.

No intermediate .txt files are created.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional


@dataclass(frozen=True)
class HFScanResult:
    chars: set[str]
    total_chars: int


def iter_hf_text_rows(
    *,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-103-raw-v1",
    split: str = "train",
    text_column: str = "text",
) -> Iterator[str]:
    """Yield text rows from a Hugging Face dataset split (streaming)."""
    try:
        from datasets import load_dataset  # imported here to keep module import lightweight
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing dependency `datasets`. Install it with: pip install datasets"
        ) from e

    ds = load_dataset(dataset_name, dataset_config, split=split, streaming=True)
    for row in ds:
        value = row.get(text_column)
        if value is None:
            continue
        if isinstance(value, str):
            yield value
        else:
            yield str(value)


def scan_hf_for_charset_and_count(
    *,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-103-raw-v1",
    split: str = "train",
    text_column: str = "text",
    row_separator: str = "\n",
    max_rows: Optional[int] = None,
) -> HFScanResult:
    """Streaming scan to collect unique characters and total character count.

    This is needed for:
    - building a CharTokenizer vocab
    - computing len(loader) for avg_loss without changing the training loop body

    `max_rows` is optional for quick debugging.
    """
    chars: set[str] = set()
    total_chars = 0

    for i, row in enumerate(
        iter_hf_text_rows(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            split=split,
            text_column=text_column,
        )
    ):
        row_with_sep = row + row_separator
        chars.update(row_with_sep)
        total_chars += len(row_with_sep)

        if max_rows is not None and (i + 1) >= max_rows:
            break

    return HFScanResult(chars=chars, total_chars=total_chars)


def stream_token_ids_from_hf(
    *,
    tokenizer,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-103-raw-v1",
    split: str = "train",
    text_column: str = "text",
    row_separator: str = "\n",
) -> Iterator[int]:
    """Stream token IDs from Hugging Face text rows using an existing CharTokenizer."""
    for row in iter_hf_text_rows(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split=split,
        text_column=text_column,
    ):
        for token_id in tokenizer.encode(row + row_separator):
            yield token_id
