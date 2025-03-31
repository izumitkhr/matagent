import os
from pathlib import Path

import torch

from .dataset import CrystDataset
from .util import get_scaler_from_data_list
from .dataloader import get_dataloader


def retrieve_dataloader(cfg):
    dataset_path = Path(cfg.data.data_path)
    splits = {}
    if os.path.exists(dataset_path / "train.pt"):
        for split in ["train", "val", "test"]:
            splits[split] = torch.load(dataset_path / f"{split}.pt")

        lattice_scaler = splits["train"].lattice_scaler
        scaler = splits["train"].scaler
    else:
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)

        for split in ["train", "val", "test"]:
            splits[split] = CrystDataset(cfg.data, split=split)

        lattice_scaler = get_scaler_from_data_list(
            splits["train"].cached_data, key="scaled_lattice"
        )
        scaler = get_scaler_from_data_list(
            splits["train"].cached_data, key=splits["train"].prop
        )

        for split, dataset in splits.items():
            dataset.lattice_scaler, dataset.scaler = lattice_scaler, scaler
            torch.save(dataset, dataset_path / f"{split}.pt")

    dataloaders = get_dataloader(splits, cfg.data.batch_size)
    return dataloaders, lattice_scaler, scaler