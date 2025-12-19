# Step 5 — Training

Goal: train the GPT model on `data/cleaned/train.txt`.

Run:
- Fresh: `python step5_training/train.py`
- Resume: `python step5_training/train.py --resume`

MLOps outputs:
- `../checkpoints/epoch_{epoch}.pt` (saved every epoch)
- `config.json` (hyperparameters, written at start)
- `training_log.csv` (epoch loss)
- `tokenizer.pkl` (persisted tokenizer used by generation)
- `gpt_char_model.pt` (final weights)
