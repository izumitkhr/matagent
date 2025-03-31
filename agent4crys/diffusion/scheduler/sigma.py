import numpy as np
import torch

from .base import BaseScheduler
from ..transition.wrapped_normal import d_log_p_wrapped_normal


def sigma_norm(sigma, T=1.0, sn=10000):
    sigmas = sigma[None, :].repeat(sn, 1)
    x_sample = sigma * torch.randn_like(sigmas)
    x_sample = x_sample % T
    normal_ = d_log_p_wrapped_normal(x_sample, sigmas, T=T)
    return (normal_**2).mean(dim=0)


class SigmaScheduler(BaseScheduler):
    def __init__(
        self,
        timesteps,
        sigma_begin=0.01,
        sigma_end=1.0,
    ):
        super().__init__(timesteps=timesteps)
        self.sigma_begin = sigma_begin
        self.sigma_end = sigma_end

        sigmas = torch.tensor(
            np.exp(np.linspace(np.log(sigma_begin), np.log(sigma_end), timesteps))
        ).float()
        sigmas_norm_ = sigma_norm(sigmas)

        self.register_buffer("sigmas", torch.cat([torch.zeros([1]), sigmas], dim=0))
        self.register_buffer(
            "sigmas_norm", torch.cat([torch.ones([1]), sigmas_norm_], dim=0)
        )
