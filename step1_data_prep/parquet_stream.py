"""Parquet streaming utilities.

This file implements a minimal, educational, *streaming* Parquet -> text -> token-ID pipeline.

Key idea:
- Use pyarrow to read Parquet in batches (do not load the full dataset).
- Extract a text column (default: "text").
- Optionally scan once to build a character vocabulary for CharTokenizer.

This keeps the rest of the project (tokenizer/dataset/training/model) unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import pyarrow as pa
import pyarrow.parquet as pq


PathLike = Union[str, Path]


def _expand_parquet_paths(paths: Sequence[PathLike]) -> List[Path]:
    out: List[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            out.extend(sorted(path.glob("*.parquet")))
        else:
            out.append(path)
    # De-dup while preserving order
    seen = set()
    uniq: List[Path] = []
    for p in out:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def iter_parquet_text_batches(
    parquet_paths: Sequence[PathLike],
    *,
    text_column: str = "text",
    batch_size: int = 1024,
) -> Iterator[List[str]]:
    """Yield lists of text rows from one or more Parquet files.

    Streaming guarantee: reads row batches using pyarrow; does not materialize the full dataset.

    Notes:
    - Null values are skipped.
    - Non-string values are converted to string.
    """
    paths = _expand_parquet_paths(parquet_paths)
    if not paths:
        raise FileNotFoundError("No Parquet files found.")

    for path in paths:
        pf = pq.ParquetFile(path)

        # iter_batches yields RecordBatch objects.
        for rb in pf.iter_batches(batch_size=batch_size, columns=[text_column]):
            col = rb.column(0)

            # Convert column to Python list. This materializes only the current batch.
            values = col.to_pylist()
            out: List[str] = []
            for v in values:
                if v is None:
                    continue
                if isinstance(v, str):
                    out.append(v)
                else:
                    out.append(str(v))
            if out:
                yield out


@dataclass(frozen=True)
class ParquetScanResult:
    """Result of a streaming scan over Parquet text."""

    chars: set[str]
    total_chars: int


def scan_parquet_for_charset_and_count(
    parquet_paths: Sequence[PathLike],
    *,
    text_column: str = "text",
    batch_size: int = 1024,
    row_separator: str = "\n",
) -> ParquetScanResult:
    """One streaming pass to collect unique characters and total character count.

    Why this exists:
    - CharTokenizer needs a set of characters to build vocab.
    - Training code often wants to know approximately how many samples exist.

    This scan is still streaming: it does not load the full dataset into memory.
    """
    chars: set[str] = set()
    total_chars = 0

    for batch in iter_parquet_text_batches(
        parquet_paths, text_column=text_column, batch_size=batch_size
    ):
        for row in batch:
            # Add a separator to avoid concatenating last/first chars of adjacent rows.
            row_with_sep = row + row_separator
            chars.update(row_with_sep)
            total_chars += len(row_with_sep)

    return ParquetScanResult(chars=chars, total_chars=total_chars)


def stream_token_ids_from_parquet(
    parquet_paths: Sequence[PathLike],
    *,
    tokenizer,
    text_column: str = "text",
    parquet_batch_size: int = 1024,
    row_separator: str = "\n",
) -> Iterator[int]:
    """Stream token IDs from Parquet using an existing CharTokenizer.

    Tokenization happens on-the-fly as text batches are read.
    """
    for batch in iter_parquet_text_batches(
        parquet_paths, text_column=text_column, batch_size=parquet_batch_size
    ):
        for row in batch:
            ids = tokenizer.encode(row + row_separator)
            for i in ids:
                yield i
