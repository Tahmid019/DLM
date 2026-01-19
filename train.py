import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
from pathlib import Path

from config import LLaDAConfig
from model import LLaDA
from dataset import TextDataset

# --------------------
# Setup
# --------------------
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

tokenizer = AutoTokenizer.from_pretrained("./tokenizer")
MASK = tokenizer.mask_token_id
print("Vocab size:", len(tokenizer))

dataset = TextDataset("data/processed.txt", tokenizer, seq_len=6)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# --------------------
# Model
# --------------------
config = LLaDAConfig(vocab_size=len(tokenizer))
model = LLaDA(config).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

ckpt = CHECKPOINT_DIR / "model.pt"
if ckpt.exists():
    print("Loading checkpoint...")
    model.load_state_dict(torch.load(ckpt, map_location=device))

# --------------------
# Training step
# --------------------
def train_step(model, batch):
    model.train()
    x = batch.to(device)

    B, L = x.shape
    t = torch.rand(B, device=device).clamp(min=0.05)  # avoid 1/t explosion

    mask = torch.rand(B, L, device=device) < t.unsqueeze(1)

    x_masked = x.clone()
    x_masked[mask] = MASK

    logits = model(x_masked)

    logits = logits.view(-1, logits.size(-1))
    targets = x.view(-1)
    mask = mask.view(-1)

    loss = (
        F.cross_entropy(
            logits[mask],
            targets[mask],
            reduction="none"
        ) / t.repeat_interleave(L)[mask]
    ).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

# --------------------
# Training loop
# --------------------
for epoch in range(5):
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}")

    for batch in pbar:
        loss = train_step(model, batch)
        total_loss += loss
        pbar.set_postfix(loss=f"{loss:.4f}")

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), CHECKPOINT_DIR / "model.pt")

# ------
model.save_pretrained("llada")
tokenizer.save_pretrained("llada")
