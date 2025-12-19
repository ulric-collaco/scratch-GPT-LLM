# From-scratch Character-level GPT (PyTorch)

This repo is organized as a **step-by-step college ML project** (aimed at a 2nd-year engineering student). Each folder is one phase in the pipeline, from raw text to a trained character-level GPT and a small generation script.

## Folder structure

```
LLM/
├── data/
│   ├── raw/                # original unprocessed datasets (.txt)
│   └── cleaned/            # cleaned + split outputs (train/val)
│
├── step1_data_prep/        # merge → clean → split
├── step2_tokenizer/        # character tokenizer
├── step3_dataset/          # (x,y) next-token dataset
├── step4_model/            # embeddings/attention/block/GPT
├── step5_training/         # train script + saved artifacts
├── step6_generation/       # generate script
│
├── checkpoints/            # epoch_{epoch}.pt
└── tests/unit_tests/       # small shape/sanity scripts
```

## What each step does

### Step 1 — Data prep (`step1_data_prep/`)

Goal: convert several raw `.txt` files into training-ready `train.txt` and `val.txt`.

Scripts:
- `merge_datasets.py`: merges every `data/raw/*.txt` into `data/cleaned/merged_raw.txt`
- `clean_text.py`: minimal cleaning (whitespace normalization) into `data/cleaned/merged_clean.txt`
- `split_train_val.py`: splits `merged_clean.txt` into `data/cleaned/train.txt` and `data/cleaned/val.txt`

### Step 2 — Tokenizer (`step2_tokenizer/`)

Goal: build a **character-level** tokenizer.

- `tokenizer.py`: `CharTokenizer` maps characters → integer IDs (`stoi`) and back (`itos`).

### Step 3 — Dataset (`step3_dataset/`)

Goal: turn a stream of token IDs into training pairs for next-token prediction.

- `dataset.py`: `TextDataset` returns:
  - input `x`: a window of length `seq_len`
  - target `y`: the same window shifted by 1 token

### Step 4 — Model (`step4_model/`)

Goal: implement a small GPT-style transformer for character-level language modeling.

- `embeddings.py`: token + positional embeddings
- `attention.py`: causal self-attention (single-head and multi-head)
- `transformer_block.py`: transformer block (pre-layernorm + residual)
- `gpt.py`: stacks blocks into GPT and provides `generate()`

### Step 5 — Training (`step5_training/`)

Goal: train GPT on `data/cleaned/train.txt`.

MLOps features:
- Saves `checkpoints/epoch_{epoch}.pt` at the end of every epoch
- Writes `step5_training/config.json` at training start
- Appends epoch loss to `step5_training/training_log.csv`
- Saves tokenizer to `step5_training/tokenizer.pkl`
- Saves final weights to `step5_training/gpt_char_model.pt`
- Supports resume: `--resume` loads latest checkpoint and continues

### Step 6 — Generation (`step6_generation/`)

Goal: load trained weights + tokenizer and sample text.

## Datasets (`data/`)

- Put original text files in `data/raw/`.
- Cleaned outputs used for training live in `data/cleaned/`.

This repo currently expects plain text sources and includes:
- `data/raw/wizofoz.txt`
- `data/raw/romju.txt`
- `data/raw/kjamesbible.txt`

## Quick start

### 0) Install deps

`pip install -r requirements.txt`

### 1) Prepare data (optional)

If `data/cleaned/train.txt` already exists, you can skip this.

- `python step1_data_prep/merge_datasets.py`
- `python step1_data_prep/clean_text.py`
- `python step1_data_prep/split_train_val.py`

### 2) Train

- Fresh: `python step5_training/train.py`
- Resume: `python step5_training/train.py --resume`

### 3) Generate

- `python step6_generation/generate.py`

## Notes

- This is a **character-level** GPT: it predicts the next character given a context window.
- The code is intentionally minimal and educational (readable over optimized).
