import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import LLaDA
from dataset import TextDataset
from transformers import AutoTokenizer
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device: ", device)

tokenizer = AutoTokenizer.from_pretrained("./tokenizer")
MASK = tokenizer.mask_token_id

dataset = TextDataset("data/processed.txt", tokenizer, seq_len=6)
assert len(dataset) > 0, "Dataset is empty"
loader = DataLoader(dataset, batch_size=8, shuffle=True)

model = LLaDA(len(tokenizer)).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

def train_step(model, batch, optimizer, tokenizer):
    model.train()
    x = batch.to(next(model.parameters()).device)

    B, L = x.shape
    t = torch.rand(B, device=x.device)   # t ~ U(0,1)
    mask = torch.rand(B, L, device=x.device) < t.unsqueeze(1)

    x_masked = x.clone()
    x_masked[mask] = tokenizer.mask_token_id

    logits = model(x_masked)

    # flatten
    logits = logits.view(-1, logits.size(-1))
    targets = x.view(-1)
    mask = mask.view(-1)

    # 1/t weighting
    loss = (
        F.cross_entropy(logits[mask], targets[mask], reduction="none")
        / t.repeat_interleave(L)[mask]
    ).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


for epoch in range(5):
    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}", leave=True)

    for batch in pbar:
        loss = train_step(model, batch, optimizer, tokenizer)
        total_loss += loss

        pbar.set_postfix(loss=f"{loss:.4f}")

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")




torch.save(model.state_dict(), "checkpoints/model.pt")
