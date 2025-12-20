import torch
import torch.nn as nn

class LLaDA(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_layers=6, n_heads=8, max_len=512):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        B, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)

        x = self.token_emb(input_ids) + self.pos_emb(pos)
        x = self.transformer(x)          
        logits = self.lm_head(x)
        return logits
