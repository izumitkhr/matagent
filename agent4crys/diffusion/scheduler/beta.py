import math

import torch

from .base import BaseScheduler


def cosine_beta_schedule(timesteps, s=0.008):
    """cosine schedule as proposed in https://arxiv.org/abs/2102.09672"""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps, beta_start, beta_end):
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps, beta_start, beta_end):
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps, beta_start, beta_end):
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


class BetaScheduler(BaseScheduler):
    def __init__(
        self,
        timesteps,
        scheduler_mode,
        beta_start=0.0001,
        beta_end=0.02,
        s=0.008,
    ):
        super().__init__(timesteps=timesteps)
        if scheduler_mode == "cosine":
            betas = cosine_beta_schedule(timesteps, s)
        elif scheduler_mode == "linear":
            betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        elif scheduler_mode == "quadratic":
            betas = quadratic_beta_schedule(timesteps, beta_start, beta_end)
        elif scheduler_mode == "sigmoid":
            betas = sigmoid_beta_schedule(timesteps, beta_start, beta_end)
        else:
            raise ValueError(f"Unknown scheduler mode: {scheduler_mode}.")

        betas = torch.cat([torch.zeros([1]), betas], dim=0)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)

        sigmas = torch.zeros_like(betas)
        sigmas[1:] = (
            betas[1:] * (1.0 - alphas_cumprod[:-1]) / (1.0 - alphas_cumprod[1:])
        )
        sigmas = torch.sqrt(sigmas)

        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_1_min_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sigmas", sigmas)
        self.register_buffer("c0", sqrt_alphas_cumprod)
        self.register_buffer("c1", sqrt_1_min_alphas_cumprod)


class BetaSchedulerCat(BetaScheduler):
    def __init__(
        self,
        timesteps,
        scheduler_mode,
        beta_start=0.0001,
        beta_end=0.02,
    ):
        super().__init__(
            timesteps=timesteps,
            scheduler_mode=scheduler_mode,
            beta_start=beta_start,
            beta_end=beta_end,
        )

        self.register_buffer("log_alphas", torch.log(self.alphas))
        self.register_buffer("log_1_min_alphas", torch.log(1.0 - self.alphas))
        self.register_buffer("log_cumprod_alphas", torch.log(self.alphas_cumprod))
        self.register_buffer(
            "log_1_min_cumprod_alphas", torch.log(1.0 - self.alphas_cumprod)
        )
