# Step 4 — Model

Goal: implement a small GPT-style transformer for character-level language modeling.

Files:
- `embeddings.py`: token + positional embeddings
- `attention.py`: causal self-attention (single-head and multi-head)
- `transformer_block.py`: transformer block (pre-layernorm + residual)
- `gpt.py`: stacks blocks into a GPT model and includes a simple `generate()` method

These components are split for readability; behavior matches the original single-file implementation.
