from .time import SinusoidalTimeEmbedding, LearnableEmedding


def get_time_embedding(cfg):
    type = cfg.time_embedding.type
    dim = cfg.time_embedding.dim
    timesteps = cfg.timesteps

    if type == "sinusoid":
        return SinusoidalTimeEmbedding(dim)
    elif type == "learnable":
        return LearnableEmedding(dim, timesteps)
    else:
        raise NotImplementedError(type)
