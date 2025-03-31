import math
import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class LearnableEmedding(nn.Module):
    def __init__(self, dim, timesteps):
        super().__init__()
        num_embeddings = timesteps + 1
        self.emb = nn.Embedding(num_embeddings, dim)

    def forward(self, times):
        embeddings = self.emb(times)
        return embeddings
