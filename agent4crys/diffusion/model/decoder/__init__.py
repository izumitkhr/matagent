from .cspnet import CSPNet


def get_decoder(cfg):
    cfg_dec = cfg.decoder
    if cfg.type == "crystal_diffusion" and cfg_dec.type == "cspnet":
        return CSPNet(
            hidden_dim=cfg_dec.hidden_dim,
            latent_dim=cfg.time_embedding.dim,
            num_layers=cfg_dec.num_layers,
            act_fn=cfg_dec.act_fn,
            dis_emb=cfg_dec.dis_emb,
            num_freqs=cfg_dec.num_freqs,
            edge_style=cfg_dec.edge_style,
            ln=cfg_dec.ln,
            ip=cfg_dec.ip,
        )
    else:
        raise NotImplementedError(
            f"Unknown model and decoder combination: {cfg.type} and {cfg_dec.type}."
        )
