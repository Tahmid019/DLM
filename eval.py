import torch
from model import LLaDA
from transformers import AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("./tokenizer")
model = LLaDA(len(tokenizer)).to(device)
model.load_state_dict(torch.load("checkpoints/model.pt"))
model.eval()

prompt = "Diffusion models"
ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

print("Prompt:", prompt)
