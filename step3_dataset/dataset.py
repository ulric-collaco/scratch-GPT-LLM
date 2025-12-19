import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, tokens, seq_len: int):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self) -> int:
        # how many samples we can make
        return len(self.tokens) - self.seq_len

    def __getitem__(self, idx: int):
        x = self.tokens[idx : idx + self.seq_len]
        y = self.tokens[idx + 1 : idx + self.seq_len + 1]

        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
