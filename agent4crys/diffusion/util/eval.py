from pathlib import Path

import numpy as np
import torch

from ..model import retrieve_model


def load_model(model_path):
    if not isinstance(model_path, Path):
        model_path = Path(model_path)
    file_path = model_path / "best_model.pth"
    cfg = torch.load(file_path)["cfg"]
    model = retrieve_model(cfg)
    state_dict = torch.load(file_path)["model_state_dict"]
    model.load_state_dict(state_dict)
    lattice_scaler = torch.load(file_path)["lattice_scaler"]
    scaler = torch.load(file_path)["scaler"]
    model.lattice_scaler, model.scaler = lattice_scaler, scaler
    return model, cfg


def lattices_to_params_shape(lattices):

    lengths = torch.sqrt(torch.sum(lattices**2, dim=-1))
    angles = torch.zeros_like(lengths)
    for i in range(3):
        j = (i + 1) % 3
        k = (i + 2) % 3
        angles[..., i] = torch.clamp(
            torch.sum(lattices[..., j, :] * lattices[..., k, :], dim=-1)
            / (lengths[..., j] * lengths[..., k]),
            -1.0,
            1.0,
        )
    angles = torch.arccos(angles) * 180.0 / np.pi

    return lengths, angles
