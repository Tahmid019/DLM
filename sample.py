import torch
from model import LLaDA
from transformers import AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("./tokenizer")
MASK = tokenizer.mask_token_id

model = LLaDA(len(tokenizer)).to(device)
model.load_state_dict(torch.load("checkpoints/model.pt"))
model.eval()

L = 50
steps = 10

x = torch.full((1, L), MASK, device=device)

for step in range(steps):
    logits = model(x)
    probs = logits.softmax(-1)
    pred = probs.argmax(-1)

    conf = probs.max(-1).values
    k = int((step + 1) / steps * L)

    _, idx = conf.topk(k, largest=True)
    x[0, idx] = pred[0, idx]

text = tokenizer.decode(x[0], skip_special_tokens=True)
print(text)
