import torch
import torch.nn as nn
import torch.nn.functional as F


class ContinuousTransition(nn.Module):
    def __init__(self, scheduler, num_classes):
        super().__init__()
        self.scheduler = scheduler
        self.num_classes = torch.tensor(num_classes)

    def onehot_encode(self, v):
        return F.one_hot(v, self.num_classes).float()

    def sample_noise(self, v):
        # subtract center of gravity?
        return torch.randn_like(v)

    def add_noise(self, v, timestep, num_atoms=None):
        c0 = self.scheduler.c0[timestep]
        c1 = self.scheduler.c1[timestep]
        if v.ndim == 3:
            rand_v = self.sample_noise(v)
            v_t = c0[:, None, None] * v + c1[:, None, None] * rand_v
        elif v.ndim == 1:
            v = self.onehot_encode(v)
            rand_v = self.sample_noise(v)
            v_t = (
                c0.repeat_interleave(num_atoms)[:, None] * v
                + c1.repeat_interleave(num_atoms)[:, None] * rand_v
            )
        else:
            raise Exception(f"ndim=={v.ndim}")
        return v_t, rand_v

    def get_mse_loss(self, pred_v, tar_v):
        return F.mse_loss(pred_v, tar_v)

    def sample_init(self, size):
        """Initialize with standard normal distribution"""
        return torch.randn(size).to(self.scheduler.alphas.device)

    def predict(self, v_t, v_pred, t):
        rand_v = self.sample_noise(v_t) if t > 1 else torch.zeros_like(v_t)

        alpha = self.scheduler.alphas[t]
        alpha_cumprod = self.scheduler.alphas_cumprod[t]
        sigma = self.scheduler.sigmas[t]

        c0 = 1.0 / torch.sqrt(alpha)
        c1 = (1.0 - alpha) / torch.sqrt(1.0 - alpha_cumprod)

        return c0 * (v_t - c1 * v_pred) + sigma * rand_v
