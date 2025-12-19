# From-scratch Character-level GPT (PyTorch)

This repo is organized as a **step-by-step college ML project**. Each folder is a stage in the pipeline, from raw text files to a trained character-level GPT and a simple text generator.

## Project layout

- `data/` — datasets (raw and cleaned)
- `step1_data_prep/` — merge → clean → split into train/val
- `step2_tokenizer/` — character-level tokenizer
- `step3_dataset/` — next-token dataset for training
- `step4_model/` — GPT model components (embeddings, attention, blocks, GPT)
- `step5_training/` — training script + saved run artifacts
- `step6_generation/` — generation script
- `checkpoints/` — per-epoch checkpoints (`epoch_{epoch}.pt`)
- `tests/unit_tests/` — small shape/sanity scripts

## Quick start

### 0) Install deps

Create an environment and install:

- `pip install -r requirements.txt`

### 1) Prepare data (optional)

Raw `.txt` files live in `data/raw/`. Cleaned outputs go to `data/cleaned/`.

- Merge raw files:
  - `python step1_data_prep/merge_datasets.py`
- Clean merged text (whitespace normalization):
  - `python step1_data_prep/clean_text.py`
- Split into train/val:
  - `python step1_data_prep/split_train_val.py`

If `data/cleaned/train.txt` already exists, you can skip this.

### 2) Train

- Fresh training:
  - `python step5_training/train.py`
- Resume training from latest checkpoint:
  - `python step5_training/train.py --resume`

Artifacts written by training:
- `checkpoints/epoch_{epoch}.pt` (every epoch)
- `step5_training/config.json` (hyperparameters)
- `step5_training/training_log.csv` (epoch loss)
- `step5_training/tokenizer.pkl` (persisted tokenizer)
- `step5_training/gpt_char_model.pt` (final weights)

### 3) Generate

- `python step6_generation/generate.py`

This loads `step5_training/tokenizer.pkl` so the vocab matches training.

## Notes

- This is a **character-level** GPT: it predicts the next character given a context window.
- The goal is clarity and learning; the code favors readability over optimizations.
