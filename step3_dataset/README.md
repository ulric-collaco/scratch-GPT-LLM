# Step 3 — Dataset

Goal: convert a stream of token IDs into training examples for next-token prediction.

- `dataset.py` defines `TextDataset`.
  - Input `x`: a window of length `seq_len`
  - Target `y`: the same window shifted by one token

Run (from repo root):
- `python step3_dataset/test_dataset.py`
