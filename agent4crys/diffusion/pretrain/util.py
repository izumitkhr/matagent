import torch

import pandas as pd
from pymatgen.core.structure import Structure, Lattice

from agent4crys.diffusion.util.eval import lattices_to_params_shape


def diffusion(loader, model, device):
    frac_coords = []
    num_atoms = []
    atom_types = []
    lattices = []
    traj_all = []
    for batch in loader:
        batch = batch.to(device)
        outputs, traj = model.sample(batch)
        frac_coords.append(outputs["frac_coords"].detach().cpu())
        num_atoms.append(outputs["num_atoms"].detach().cpu())
        atom_types.append(outputs["atom_types"].detach().cpu())
        lattices.append(outputs["lattices"].detach().cpu())
        traj_all.append(traj)
    frac_coords = torch.cat(frac_coords, dim=0)
    num_atoms = torch.cat(num_atoms, dim=0)
    atom_types = torch.cat(atom_types, dim=0)
    lattices = torch.cat(lattices, dim=0)
    lengths, angles = lattices_to_params_shape(lattices)
    # gen_mat = (frac_coords, atom_types, lattices, lengths, angles, num_atoms, traj_all)
    gen_mat = (frac_coords, atom_types, lengths, angles, num_atoms)
    return gen_mat


def get_crystals_df(frac_coords, atom_types, lengths, angles, num_atoms):
    assert frac_coords.size(0) == atom_types.size(0) == num_atoms.sum()
    assert lengths.size(0) == angles.size(0) == num_atoms.size(0)

    start_idx = 0
    cifs = []
    for batch_idx, num_atoms in enumerate(num_atoms.tolist()):
        cur_frac_coords = frac_coords.narrow(0, start_idx, num_atoms)
        cur_atom_types = atom_types.narrow(0, start_idx, num_atoms)
        cur_lengths = lengths[batch_idx]
        cur_angles = angles[batch_idx]

        crystal = Structure(
            lattice=Lattice.from_parameters(
                *(cur_lengths.tolist() + cur_angles.tolist())
            ),
            species=cur_atom_types,
            coords=cur_frac_coords,
            coords_are_cartesian=False,
        )
        cif = crystal.to(fmt="cif")
        cifs.append(cif)
        # crystal_array_list.append(
        #     {
        #         "frac_coords": cur_frac_coords.detach().cpu().numpy(),
        #         "atom_types": cur_atom_types.detach().cpu().numpy(),
        #         "lengths": cur_lengths.detach().cpu().numpy(),
        #         "angles": cur_angles.detach().cpu().numpy(),
        #     }
        # )
        start_idx = start_idx + num_atoms
    # TODO: update dummies !
    df = pd.DataFrame(
        {
            "cif": cifs,
            "formation_energy_per_atom": [0.0] * len(cifs),  # dummy
            "band_gap": [0.0] * len(cifs),  # dummy
        }
    )
    return df
