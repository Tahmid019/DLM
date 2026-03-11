import torch
from torch.utils.data import Dataset
from pathlib import Path

class LLaDAPreparedChunkDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.files = sorted(self.root_dir.glob("processed_chunk_*.pt"))
        assert len(self.files) > 0, "No processed_chunk_*.pt files found"

        self.index = []
        for f_id, f in enumerate(self.files):
            data = torch.load(f, map_location="cpu")
            for i in range(len(data)):
                self.index.append((f_id, i))

        print(f"[Dataset] {len(self.index)} samples from {len(self.files)} files")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        f_id, i = self.index[idx]
        if not hasattr(self, "_cache") or self._cache_id != f_id:
            self._cache = torch.load(self.files[f_id], map_location="cpu")
            self._cache_id = f_id
        return self._cache[i]
