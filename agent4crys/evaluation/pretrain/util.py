import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from agent4crys.evaluation.models.util.model import retrieve_model
from agent4crys.evaluation.models.util.data import get_pyg_dataset


def load_model(model_path):
    if not isinstance(model_path, Path):
        model_path = Path(model_path)
    file_path = model_path / "best_model.pth"
    cfg = torch.load(file_path)["cfg"]
    model = retrieve_model(cfg)
    state_dict = torch.load(file_path)["model_state_dict"]
    model.load_state_dict(state_dict)
    std_train = torch.load(file_path)["std_train"]
    mean_train = torch.load(file_path)["mean_train"]
    return model, cfg, std_train, mean_train


def load_evaluation_model(eval_prop="formation_energy_per_atom"):
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    if eval_prop == "formation_energy_per_atom":
        model_path = current_dir / "comformer"
        model, cfg, std_train, mean_train = load_model(model_path)
        return model, cfg, std_train, mean_train
    else:
        raise ValueError(f"Model type {eval_prop} is not supported.")


def get_dataloader(mat_df, cfg, std_train, mean_train):
    dataset, _, _ = get_pyg_dataset(
        mat_df,
        target=cfg.data.target,
        neighbor_strategy=cfg.data.neighbor_strategy,
        atom_features=cfg.data.atom_features,
        use_canonize=cfg.data.use_canonize,
        line_graph=True,
        cutoff=cfg.data.cutoff,
        max_neighbors=cfg.data.max_neighbors,
        use_lattice=cfg.data.use_lattice,
        use_angle=False,
        mean_train=mean_train,
        std_train=std_train,
        eval=True,
    )
    collate_fn = dataset.collate_line_graph
    loader = DataLoader(
        dataset, batch_size=len(dataset), shuffle=False, collate_fn=collate_fn
    )
    return loader, loader.dataset.prepare_batch
