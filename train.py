import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import LLaDA
from dataset import TextDataset
from transformers import AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("./tokenizer")
MASK = tokenizer.mask_token_id

text = open("data/processed.txt").read()
dataset = TextDataset(text, tokenizer, seq_len=32)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

model = LLaDA(len(tokenizer)).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for epoch in range(5):
    for x in loader:
        x = x.to(device)

        t = torch.rand(1).item()
        mask = torch.rand_like(x.float()) < t
        x_t = x.clone()
        x_t[mask] = MASK

        logits = model(x_t)

        loss = F.cross_entropy(
            logits[mask],
            x[mask],
            reduction="mean"
        ) / t

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "checkpoints/model.pt")
