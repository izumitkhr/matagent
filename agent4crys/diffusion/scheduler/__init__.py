from .beta import BetaScheduler, BetaSchedulerCat
from .sigma import SigmaScheduler


def get_scheduler(cfg, timesteps):
    if cfg.type == "wn":
        return SigmaScheduler(
            timesteps=timesteps,
            sigma_begin=cfg.sigma_begin,
            sigma_end=cfg.sigma_end,
        )
    elif cfg.type == "continuous":
        return BetaScheduler(
            timesteps=timesteps,
            scheduler_mode=cfg.mode,
            beta_start=cfg.beta_start,
            beta_end=cfg.beta_end,
        )
    elif cfg.type == "categorical" or cfg.type == "general_categorical":
        return BetaSchedulerCat(
            timesteps=timesteps,
            scheduler_mode=cfg.mode,
            beta_start=cfg.beta_start,
            beta_end=cfg.beta_end,
        )
    else:
        raise ValueError(f"Unknown scheduler type: {cfg.type}.")
