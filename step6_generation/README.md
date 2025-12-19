# Step 6 — Generation

Goal: load the trained model and sample text continuation.

Run:
- `python step6_generation/generate.py`

This script loads:
- `../step5_training/tokenizer.pkl`
- `../step5_training/gpt_char_model.pt`

so inference uses the exact same tokenizer as training.
