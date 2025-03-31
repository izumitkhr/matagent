import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from .util import preprocess, add_scaled_lattice_prop


class CrystDataset(Dataset):
    def __init__(self, cfg, split="train"):
        super().__init__()
        self.path = f"{cfg.root_path}/{split}.csv"
        self.df = pd.read_csv(self.path)
        self.name = split
        self.prop = cfg.prop
        self.niggli = cfg.niggli
        self.primitive = cfg.primitive
        self.graph_method = cfg.graph_method

        self.cached_data = preprocess(
            self.path,
            cfg.preprocess_workers,
            niggli=self.niggli,
            primitive=self.primitive,
            graph_method=self.graph_method,
            prop_list=[cfg.prop],
        )

        add_scaled_lattice_prop(self.cached_data, cfg.lattice_scale_method)
        self.lattice_scaler = None
        self.scaler = None

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]

        # scaler is set later
        prop = self.scaler.transform(data_dict[self.prop])
        (
            frac_coords,
            atom_types,
            lengths,
            angles,
            edge_indices,
            to_jimages,
            num_atoms,
        ) = data_dict["graph_arrays"]

        data = Data(
            frac_coords=torch.Tensor(frac_coords),
            atom_types=torch.LongTensor(atom_types),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            # edge_index=torch.LongTensor(
            #     edge_indices.T
            # ).contiguous(),  # shape (2, num_edges)
            # to_jimages=torch.LongTensor(to_jimages),
            num_atoms=num_atoms,
            # num_bonds=edge_indices.shape[0],
            num_nodes=num_atoms,
            # y=prop.view(1, -1),
        )
        return data

    def __repr__(self) -> str:
        return f"CrystDataset({self.name=}, {self.path=})"
