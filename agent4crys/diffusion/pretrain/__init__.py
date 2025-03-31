import os
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from pymatgen.core.composition import Composition
from pymatgen.core.periodic_table import Element

from agent4crys.diffusion.util.eval import load_model
from agent4crys.diffusion.pretrain.util import diffusion, get_crystals_df


def load_diffusion_model(model_type="diffcsp"):
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    if model_type == "diffcsp":
        model_path = current_dir / "diffcsp"
        model, cfg = load_model(model_path)
        return model
    else:
        raise ValueError(f"Model type {model_type} is not supported.")


class CompDataset(Dataset):
    def __init__(self, input_dict, batch_size_per_Z=5, max_Z=4, max_natoms=34):
        super().__init__()
        self.reduced_composition = Composition(
            input_dict["composition"]
        ).reduced_composition
        num_atoms = self.reduced_composition.num_atoms

        if num_atoms * max_Z > max_natoms:
            max_Z = int(max_natoms // num_atoms)

        self.n_samples = batch_size_per_Z * max_Z
        self.formula_units = []
        for i in range(1, max_Z + 1):
            self.formula_units += [i] * batch_size_per_Z

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        formula_units = self.formula_units[index]
        comp = self.reduced_composition * formula_units
        cell_num_atoms = int(comp.num_atoms)

        comp_dict = comp.get_el_amt_dict()
        atom_types = [
            Element(elem).Z
            for elem, count in comp_dict.items()
            for _ in range(int(count))
        ]

        data = Data(
            num_atoms=cell_num_atoms,
            num_nodes=cell_num_atoms,
            atom_types=torch.LongTensor(atom_types),
        )
        return data


def run_diffusion_model(
    input_dict, model, model_type="diffcsp", batch_size_per_Z=5, max_Z=4
):
    dataset = CompDataset(input_dict, batch_size_per_Z=batch_size_per_Z, max_Z=max_Z)
    loader = DataLoader(dataset, batch_size=len(dataset))

    if model_type == "diffcsp":
        frac_coords, atom_types, lengths, angles, num_atoms = diffusion(
            loader, model, device=model.device
        )
        gen_mat_df = get_crystals_df(
            frac_coords, atom_types, lengths, angles, num_atoms
        )
        return gen_mat_df
    else:
        raise ValueError(f"Model type {model_type} is not supported.")
