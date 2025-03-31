import torch


def get_loss_func(cfg):
    if cfg.cost_func == "mse":
        return torch.nn.MSELoss()
    elif cfg.cost_func == "l1":
        return torch.nn.L1Loss()
    else:
        raise NotImplementedError(f"Cost function {cfg.cost_func} not implemented")
