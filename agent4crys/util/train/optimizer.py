import torch


def get_optimizer(cfg, model):
    """Instantiate an optimizer."""
    if cfg.type == "adam":
        return get_adam_optimizer(cfg, model)
    elif cfg.type == "adamW":
        return get_adamW_optimizer(cfg, model)
    elif cfg.type == "sgd":
        return get_sgd_optimizer(cfg, model)
    else:
        raise NotImplementedError(f"Optimizer type {cfg.type} is not supported.")


def get_adam_optimizer(cfg, model):
    return torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=cfg.betas,
        eps=float(cfg.eps),
    )


def get_adamW_optimizer(cfg, model):
    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=cfg.betas,
        eps=float(cfg.eps),
    )


def get_sgd_optimizer(cfg, model):
    return torch.optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=0.9,
        weight_decay=cfg.weight_decay,
    )
