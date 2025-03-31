from .model import CrystalDiffusion


def retrieve_model(cfg):
    if cfg.model.type == "crystal_diffusion":
        return CrystalDiffusion(cfg.model)
    else:
        raise NotImplementedError(f"Unknown model type: {cfg.model.type}")
