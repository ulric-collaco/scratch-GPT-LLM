# Step 1 — Data preparation

Goal: turn a few raw `.txt` files into training-ready `train.txt` and `val.txt`.

Scripts:
- `merge_datasets.py`: merges every `data/raw/*.txt` into `data/cleaned/merged_raw.txt`
- `clean_text.py`: normalizes whitespace into `data/cleaned/merged_clean.txt`
- `split_train_val.py`: splits `merged_clean.txt` into `data/cleaned/train.txt` and `data/cleaned/val.txt`

Run (from repo root):
- `python step1_data_prep/merge_datasets.py`
- `python step1_data_prep/clean_text.py`
- `python step1_data_prep/split_train_val.py`
