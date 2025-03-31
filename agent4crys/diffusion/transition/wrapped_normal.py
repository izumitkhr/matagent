import torch
import torch.nn as nn
import torch.nn.functional as F


def p_wrapped_normal(x, sigma, N=10, T=1.0):
    p_ = 0
    for i in range(-N, N + 1):
        p_ += torch.exp(-((x + T * i) ** 2) / 2 / sigma**2)
    return p_


def d_log_p_wrapped_normal(x, sigma, N=10, T=1.0):
    p_ = 0
    for i in range(-N, N + 1):
        p_ += (x + T * i) / sigma**2 * torch.exp(-((x + T * i) ** 2) / 2 / sigma**2)
    return p_ / p_wrapped_normal(x, sigma, N, T)


class WNtransition(nn.Module):
    def __init__(
        self,
        scheduler,
    ):
        super().__init__()
        self.scheduler = scheduler

    def sample_noise(self, v):
        return torch.randn_like(v)

    def add_noise(self, v, timestep, num_atoms):
        sigmas = self.scheduler.sigmas[timestep]
        sigmas_norm = self.scheduler.sigmas_norm[timestep]

        sigmas_per_atom = sigmas.repeat_interleave(num_atoms)[:, None]
        # [num_nodes] -> [num_nodes, 1]
        sigmas_norm_per_atom = sigmas_norm.repeat_interleave(num_atoms)[:, None]

        rand_v = self.sample_noise(v)

        # wrap
        v_t = (v + sigmas_per_atom * rand_v) % 1.0

        tar_v = d_log_p_wrapped_normal(
            sigmas_per_atom * rand_v, sigmas_per_atom
        ) / torch.sqrt(sigmas_norm_per_atom)

        return v_t, rand_v, tar_v

    def get_mse_loss(self, pred_v, tar_v):
        return F.mse_loss(pred_v, tar_v)

    def sample_init(self, size):
        """Initialize with uniform distribution"""
        return torch.rand(size).to(self.scheduler.sigmas.device)

    def correct(self, v_t, v_pred, t, step_lr):

        rand_v = self.sample_noise(v_t) if t > 1 else torch.zeros_like(v_t)

        sigma = self.scheduler.sigmas[t]
        sigma_norm = self.scheduler.sigmas_norm[t]

        step_size = step_lr * (sigma / self.scheduler.sigma_begin) ** 2
        std = torch.sqrt(2 * step_size)

        v_pred = v_pred * torch.sqrt(sigma_norm)

        return v_t - step_size * v_pred + std * rand_v

    def predict(self, v_t, v_pred, t):

        rand_v = self.sample_noise(v_t) if t > 1 else torch.zeros_like(v_t)

        sigma = self.scheduler.sigmas[t]
        sigma_norm = self.scheduler.sigmas_norm[t]
        adjacent_sigma = self.scheduler.sigmas[t - 1]

        step_size = sigma**2 - adjacent_sigma**2
        std = torch.sqrt((adjacent_sigma**2 * (step_size)) / sigma**2)

        v_pred = v_pred * torch.sqrt(sigma_norm)

        return v_t - step_size * v_pred + std * rand_v
