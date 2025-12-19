# Datasets

## Folder layout

- `raw/` contains the original source `.txt` files.
- `cleaned/` contains processed outputs used for training:
  - `train.txt`
  - `val.txt`

## Sources

This project expects plain-text books/plays/etc. placed in `data/raw/`.

The repository currently includes:
- `wizofoz.txt`
- `romju.txt`
- `kjamesbible.txt`

## How the cleaned data is produced

Run Step 1 scripts from the repository root:

1. Merge all `data/raw/*.txt` into one file:
   - `python step1_data_prep/merge_datasets.py`
2. Clean the merged text (whitespace normalization):
   - `python step1_data_prep/clean_text.py`
3. Split into train/val:
   - `python step1_data_prep/split_train_val.py`

Outputs are written to `data/cleaned/`.
