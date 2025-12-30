import torch
import torch.nn as nn

class LLaDA(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_layers=6, n_heads=8):
        super().__init__()
        self.vocab_size = vocab_size

        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(2048, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0)
        h = self.emb(x) + self.pos(pos)
        h = self.encoder(h)
        return self.lm_head(h)
