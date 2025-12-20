import gradio as gr
import torch
from model import LLaDA
from transformers import AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("./tokenizer")
MASK = tokenizer.mask_token_id

model = LLaDA(len(tokenizer)).to(device)
model.load_state_dict(torch.load("checkpoints/model.pt"))
model.eval()

def generate(prompt, steps=10):
    L = 50
    x = torch.full((1, L), MASK, device=device)

    for _ in range(steps):
        logits = model(x)
        probs = logits.softmax(-1)
        pred = probs.argmax(-1)
        x = torch.where(x == MASK, pred, x)

    return tokenizer.decode(x[0], skip_special_tokens=True)

gr.Interface(
    fn=generate,
    inputs=["text", gr.Slider(1, 20, value=10)],
    outputs="text",
    title="LLaDA Mini (Diffusion Language Model)"
).launch()
