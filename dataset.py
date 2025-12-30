import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, text, tokenizer, seq_len=128):
        self.seq_len = seq_len
        self.tokenizer = tokenizer

        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) < 1:
            raise ValueError("Text too short")

        repeat = (seq_len // len(tokens)) + 1
        self.tokens = (tokens * repeat)

    def __len__(self):
        return 100_000  

    def __getitem__(self, idx):
        start = torch.randint(0, len(self.tokens) - self.seq_len, (1,)).item()
        return torch.tensor(self.tokens[start:start + self.seq_len])
