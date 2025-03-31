import os
import glob
import random
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from pymatgen.core.structure import Structure
from transformers import T5Tokenizer
from torch.utils.data import DataLoader

from agent4crys.util.periodic_table import chemical_element_symbols


def load_dataloaders(base_model, data_dir, batch_size, attributes):
    tokenizer = T5Tokenizer.from_pretrained(base_model)

    data_dir = Path(data_dir)
    dataset_trn = MyDataset(data_dir / "train.csv", tokenizer, attributes)
    dataset_val = MyDataset(data_dir / "val.csv", tokenizer, attributes)
    dataset_tst = MyDataset(data_dir / "test.csv", tokenizer, attributes)

    loader_trn = DataLoader(dataset_trn, batch_size=batch_size, shuffle=True)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    loader_tst = DataLoader(dataset_tst, batch_size=batch_size, shuffle=False)

    return loader_trn, loader_val, loader_tst, dataset_trn.tokenizer


class MyDataset(Dataset):
    def __init__(self, csv_fn, tokenizer, attributes, max_length=128):
        super().__init__()

        if not os.path.exists(csv_fn) and not glob.glob(csv_fn):
            raise ValueError(f"CSV file {csv_fn} does not exist")

        df = pd.read_csv(csv_fn)
        self.inputs = df.to_dict(orient="records")

        tokenizer.add_tokens(chemical_element_symbols)

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.attributes = attributes

    def get_composition(self, input_dict):
        crystal = Structure.from_str(input_dict["cif"], fmt="cif")
        comp = crystal.composition.reduced_composition
        return comp.formula

    def get_property(self, input_dict, attributes):

        all_attributes = [
            "formation_energy_per_atom",
            "band_gap",
            "e_above_hull",
            "elements",
        ]

        # sample
        num_attributes = random.randint(1, len(all_attributes))
        attributes = random.sample(all_attributes, num_attributes)

        prompt = "Below is a description of a material."

        prompt_lookup = {
            "formation_energy_per_atom": "The formation energy per atom is",
            "band_gap": "The band gap is",
            "e_above_hull": "The energy above the convex hull is",
            "elements": "The elements are",
        }

        unit_lookup = {
            "formation_energy_per_atom": "eV/atom",
            "band_gap": "eV",
            "e_above_hull": "eV/atom",
        }

        for attr in attributes:
            if attr in ["formation_energy_per_atom", "band_gap", "e_above_hull"]:
                prompt += f" {prompt_lookup[attr]} {float(input_dict[attr]):.3f} {unit_lookup[attr]}."
            elif attr == "elements":
                crystal = Structure.from_str(input_dict["cif"], fmt="cif")
                elements = list(crystal.composition.as_dict().keys())
                prompt += f" {prompt_lookup[attr]} {', '.join(elements)}."
            else:
                raise ValueError(f"Attribute {attr} not found in prompt_lookup")

        return prompt

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        input_dict = self.inputs[index]

        comp_text = self.get_composition(input_dict)
        prop_text = self.get_property(input_dict, self.attributes)

        comp_encoding = self.tokenizer(
            comp_text, truncation=True, max_length=self.max_length, padding="max_length"
        )
        prop_encoding = self.tokenizer(
            prop_text, truncation=True, max_length=self.max_length, padding="max_length"
        )

        comp_ids = torch.tensor(comp_encoding["input_ids"])
        comp_mask = torch.tensor(comp_encoding["attention_mask"])
        prop_ids = torch.tensor(prop_encoding["input_ids"])
        prop_mask = torch.tensor(prop_encoding["attention_mask"])

        return comp_ids, comp_mask, prop_ids, prop_mask
