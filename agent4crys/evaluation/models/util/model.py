from ..comformer.models.comformer import iComformer


def retrieve_model(cfg):
    if cfg.model.type == "comformer":
        return iComformer()
    else:
        raise ValueError(f"Model type {cfg.model.type} not supported")
