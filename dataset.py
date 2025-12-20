import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, text, tokenizer, seq_len=128):
        tokens = tokenizer.encode(text)
        self.data = [
            tokens[i:i+seq_len]
            for i in range(0, len(tokens) - seq_len)
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx])
