import torch.nn as nn
from transformers import T5EncoderModel


class T5Encoder(nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.t5 = T5EncoderModel.from_pretrained("t5-base").encoder
        self.t5.resize_token_embeddings(len(tokenizer))

    def forward(self, input_ids, attention_mask):
        return self.t5(input_ids, attention_mask=attention_mask).last_hidden_state


class SimCLR(nn.Module):
    def __init__(self, tokenizer, projection_dim=256):
        super().__init__()
        self.encoder = T5Encoder(tokenizer)

        self.projector = nn.Sequential(
            nn.Linear(
                self.encoder.t5.config.d_model,
                self.encoder.t5.config.d_model,
                bias=False,
            ),
            nn.ReLU(),
            nn.Linear(
                self.encoder.t5.config.d_model,
                projection_dim,
                bias=False,
            ),
        )

    def forward(self, x_i, x_j, mask_i, mask_j):
        h_i = self.encoder(x_i, mask_i)
        h_j = self.encoder(x_j, mask_j)
        z_i = self.projector(h_i)
        z_j = self.projector(h_j)

        z_i = (z_i * mask_i.unsqueeze(-1)).sum(dim=-2) / mask_i.sum(
            dim=-1, keepdim=True
        )
        z_j = (z_j * mask_j.unsqueeze(-1)).sum(dim=-2) / mask_j.sum(
            dim=-1, keepdim=True
        )

        return z_i, z_j
