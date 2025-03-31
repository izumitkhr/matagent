import torch
import torch.nn as nn
import torch.nn.functional as F


def get_loss_func(cfg):
    if cfg.cost_func == "default":
        return DefaultLoss(cfg)
    else:
        raise NotImplementedError(f"Cost function {cfg.cost_func} not implemented")


class DefaultLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.cost_lattice = cfg.cost_lattice
        self.cost_coord = cfg.cost_coord
        # self.cost_type = cfg.cost_type

    def forward(self, loss_l, loss_f, loss_t=0):
        loss = (
            self.cost_lattice * loss_l
            + self.cost_coord * loss_f
            # + self.cost_type * loss_t
        )
        return loss

    def get_current_costs(self):
        return {
            "cost_lattice": self.cost_lattice,
            "cost_coord": self.cost_coord,
            # "cost_type": self.cost_type,
        }
