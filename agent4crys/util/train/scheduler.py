import torch


def get_scheduler(cfg, optimizer, steps_per_epoch=None):
    if cfg.lr_scheduler.type == "none":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)
    if cfg.lr_scheduler.type == "plateau":
        return get_plateau_scheduler(cfg, optimizer)
    elif cfg.lr_scheduler.type == "step":
        return get_step_scheduler(cfg, optimizer)
    elif cfg.lr_scheduler.type == "exponential":
        return get_exponential_scheduler(cfg, optimizer)
    elif cfg.lr_scheduler.type == "onecycle":
        return get_onecycle_scheduler(cfg, optimizer, steps_per_epoch)
    elif cfg.lr_scheduler.type == "cosine":
        return get_cosine_scheduler(cfg, optimizer)
    else:
        raise NotImplementedError(
            f"Scheduler type {cfg.lr_scheduler.type} is not supported."
        )


def get_plateau_scheduler(cfg, optimizer):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        factor=cfg.lr_scheduler.factor,
        patience=cfg.lr_scheduler.patience,
        min_lr=float(cfg.lr_scheduler.min_lr),
    )


def get_step_scheduler(cfg, optimizer):
    return torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer,
        step_size=cfg.lr_scheduler.step_size,
        gamma=cfg.lr_scheduler.gamma,
    )


def get_exponential_scheduler(cfg, optimizer):
    return torch.optim.lr_scheduler.ExponentialLR(
        optimizer=optimizer,
        gamma=cfg.lr_scheduler.gamma,
    )


def get_onecycle_scheduler(cfg, optimizer, steps_per_epoch):
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=cfg.optimizer.lr,
        epochs=cfg.epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,
    )


def get_cosine_scheduler(cfg, optimizer, T_max=50):
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=T_max, eta_min=0.0
    )
