"""Streaming dataset utilities.

This module converts a *stream* of token IDs into (X, Y) training pairs
for next-token prediction, without needing a giant in-memory token array.

Design choice (educational + compatible):
- Provide a small loader-like class that:
  - is iterable over batches of (x, y)
  - supports __len__ so the existing training loop can compute avg_loss

This avoids changing the model or the training-loop body.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import torch

from step1_data_prep.parquet_stream import PathLike, stream_token_ids_from_parquet


class StreamingBatchLoader:
    """A minimal, single-process streaming loader that yields batches of (x, y).

    It streams token IDs from Parquet, builds sliding windows of length (seq_len + 1),
    and packs them into batches.

    Why not torch DataLoader?
    - IterableDataset + DataLoader does not provide a reliable len(loader).
    - Your current training loop uses len(loader) to compute avg loss.

    This class keeps that training loop unchanged.
    """

    def __init__(
        self,
        parquet_paths: Sequence[PathLike],
        *,
        tokenizer,
        seq_len: int,
        batch_size: int,
        total_tokens: int,
        text_column: str = "text",
        parquet_batch_size: int = 1024,
        row_separator: str = "\n",
    ) -> None:
        self.parquet_paths = list(parquet_paths)
        self.tokenizer = tokenizer
        self.seq_len = int(seq_len)
        self.batch_size = int(batch_size)
        self.total_tokens = int(total_tokens)
        self.text_column = text_column
        self.parquet_batch_size = int(parquet_batch_size)
        self.row_separator = row_separator

        # total samples for a contiguous stream is (N - seq_len)
        self.total_samples = max(0, self.total_tokens - self.seq_len)
        self.total_batches = int(math.ceil(self.total_samples / self.batch_size)) if self.total_samples else 0

    def __len__(self) -> int:
        return self.total_batches

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        # Stream token ids for one "epoch" pass.
        token_iter = stream_token_ids_from_parquet(
            self.parquet_paths,
            tokenizer=self.tokenizer,
            text_column=self.text_column,
            parquet_batch_size=self.parquet_batch_size,
            row_separator=self.row_separator,
        )

        window: Deque[int] = deque(maxlen=self.seq_len + 1)
        batch_x: List[torch.Tensor] = []
        batch_y: List[torch.Tensor] = []

        samples_emitted = 0

        for token_id in token_iter:
            window.append(int(token_id))
            if len(window) < self.seq_len + 1:
                continue

            x = torch.tensor(list(window)[: self.seq_len], dtype=torch.long)
            y = torch.tensor(list(window)[1 : self.seq_len + 1], dtype=torch.long)

            batch_x.append(x)
            batch_y.append(y)
            samples_emitted += 1

            # Slide by 1 (deque maxlen does this automatically on append).
            # But we need to ensure the next sample shifts by exactly one token.
            # With a fixed-size deque, appending already discards the oldest.

            if len(batch_x) == self.batch_size:
                yield torch.stack(batch_x, dim=0), torch.stack(batch_y, dim=0)
                batch_x.clear()
                batch_y.clear()

            if samples_emitted >= self.total_samples:
                break

        # final partial batch
        if batch_x:
            yield torch.stack(batch_x, dim=0), torch.stack(batch_y, dim=0)
