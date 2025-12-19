import pickle
from pathlib import Path

import torch

from step2_tokenizer.tokenizer import CharTokenizer
from step4_model.gpt import GPT


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Project-relative paths so this runs from any working directory.
    project_root = Path(__file__).resolve().parents[1]
    artifacts_dir = project_root / "step5_training"

    tokenizer_path = artifacts_dir / "tokenizer.pkl"
    model_path = artifacts_dir / "gpt_char_model.pt"

    # ------------------
    # MLOps: load the persisted tokenizer
    # ------------------
    # This ensures inference uses the exact same vocab mapping as training.
    try:
        with tokenizer_path.open("rb") as f:
            tokenizer = pickle.load(f)
    except FileNotFoundError:
        # Fallback for first-time runs (e.g., before training has been executed).
        train_path = project_root / "data" / "cleaned" / "train.txt"
        with train_path.open("r", encoding="utf-8") as f:
            text = f.read()
        tokenizer = CharTokenizer(text)

    # Build model (same hyperparameters as training)
    model = GPT(
        vocab_size=tokenizer.vocab_size,
        seq_len=32,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
    ).to(device)

    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Starting prompt
    prompt = "but the bullock"
    idx = torch.tensor([tokenizer.encode(prompt)], device=device)

    # Generate
    out = model.generate(idx, max_new_tokens=300, temperature=0.5)
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
