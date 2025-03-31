import numpy as np
import torch
import torch.nn as nn


class BaseScheduler(nn.Module):
    def __init__(self, timesteps):
        super().__init__()
        self.timesteps = timesteps

    def uniform_sample_t(self, batch_size, device):
        ts = np.random.choice(np.arange(1, self.timesteps + 1), batch_size)
        return torch.from_numpy(ts).to(device)
