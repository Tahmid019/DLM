import torch
import torch.nn as nn
from transformers import PreTrainedModel
from config import LLaDAConfig


class LLaDA(PreTrainedModel):
    config_class = LLaDAConfig
    base_model_prefix = "llada"

    def __init__(self, config: LLaDAConfig):
        super().__init__(config)

        self.vocab_size = config.vocab_size
        self.d_model = config.d_model

        # --------------
        self.emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos = nn.Embedding(config.max_position_embeddings, config.d_model)

        # ------------
        enc_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=4 * config.d_model,  
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,  
        )

        self.encoder = nn.TransformerEncoder(
            enc_layer,
            num_layers=config.n_layers,
            norm=nn.LayerNorm(config.d_model),
        )

        # ---------
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # ----
        self.lm_head.weight = self.emb.weight

        self.post_init()

    def forward(self, input_ids: torch.LongTensor, **kwargs):
        """
        input_ids: (B, L)
        returns logits: (B, L, vocab_size)
        """
        B, L = input_ids.shape

        pos_ids = torch.arange(
            L, device=input_ids.device
        ).unsqueeze(0)

        h = self.emb(input_ids) + self.pos(pos_ids)
        h = self.encoder(h)
        logits = self.lm_head(h)

        return logits


def count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
