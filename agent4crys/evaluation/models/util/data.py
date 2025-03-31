import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from jarvis.core.atoms import Atoms
from pandarallel import pandarallel
import torch
from torch.utils.data import DataLoader

from ..comformer.graphs import PygStructureDataset, PygGraph


def retrieve_dataloader(cfg):
    cfg = cfg.data
    data_path = Path(cfg.root_path)
    dataset_trn = data_path / "train.csv"
    dataset_val = data_path / "val.csv"
    dataset_tst = data_path / "test.csv"

    out_path = Path(cfg.data_path)
    out_file_path = out_path / "preprocessed_data.pth"
    if os.path.exists(out_file_path):
        print(f"Loading preprocessed data from {out_path}.")
        load_data = torch.load(out_file_path)
        trn_data = load_data["train"]
        val_data = load_data["val"]
        tst_data = load_data["test"]
        std_train = load_data["std_train"]
        mean_train = load_data["mean_train"]
    else:
        print(f"Saving preprocessed data to {out_file_path}.")
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        trn_data, mean_train, std_train = get_pyg_dataset(
            dataset=dataset_trn,
            target=cfg.target,
            neighbor_strategy=cfg.neighbor_strategy,
            atom_features=cfg.atom_features,
            use_canonize=cfg.use_canonize,
            line_graph=True,
            cutoff=cfg.cutoff,
            max_neighbors=cfg.max_neighbors,
            use_lattice=cfg.use_lattice,
            use_angle=False,
        )
        val_data, _, _ = get_pyg_dataset(
            dataset=dataset_val,
            target=cfg.target,
            neighbor_strategy=cfg.neighbor_strategy,
            atom_features=cfg.atom_features,
            use_canonize=cfg.use_canonize,
            line_graph=True,
            cutoff=cfg.cutoff,
            max_neighbors=cfg.max_neighbors,
            use_lattice=cfg.use_lattice,
            use_angle=False,
            mean_train=mean_train,
            std_train=std_train,
        )
        tst_data, _, _ = get_pyg_dataset(
            dataset=dataset_tst,
            target=cfg.target,
            neighbor_strategy=cfg.neighbor_strategy,
            atom_features=cfg.atom_features,
            use_canonize=cfg.use_canonize,
            line_graph=True,
            cutoff=cfg.cutoff,
            max_neighbors=cfg.max_neighbors,
            use_lattice=cfg.use_lattice,
            use_angle=False,
            mean_train=mean_train,
            std_train=std_train,
        )
        torch.save(
            {
                "train": trn_data,
                "val": val_data,
                "test": tst_data,
                "std_train": std_train,
                "mean_train": mean_train,
            },
            out_file_path,
        )

    # collate_fn = trn_data.collate
    # if cfg.line_graph:
    collate_fn = trn_data.collate_line_graph

    train_loader = DataLoader(
        trn_data,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    test_loader = DataLoader(
        tst_data,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        train_loader.dataset.prepare_batch,
        mean_train,
        std_train,
    )


def load_pyg_graphs(
    df: pd.DataFrame,
    # name: str = "dft_3d",
    neighbor_strategy: str = "k-nearest",
    cutoff: float = 8,
    max_neighbors: int = 12,
    # cachedir: Optional[Path] = None,
    use_canonize: bool = False,
    use_lattice: bool = False,
    use_angle: bool = False,
):
    """Construct crystal graphs.

    Load only atomic number node features
    and bond displacement vector edge features.

    Resulting graphs have scheme e.g.
    ```
    Graph(num_nodes=12, num_edges=156,
          ndata_schemes={'atom_features': Scheme(shape=(1,)}
          edata_schemes={'r': Scheme(shape=(3,)})
    ```
    """

    def atoms_to_graph(cif):
        """Convert structure dict to DGLGraph."""
        structure = Atoms.from_cif(
            from_string=cif, use_cif2cell=False, get_primitive_atoms=False
        )
        return PygGraph.atom_dgl_multigraph(
            structure,
            neighbor_strategy=neighbor_strategy,
            cutoff=cutoff,
            atom_features="atomic_number",
            max_neighbors=max_neighbors,
            compute_line_graph=False,
            use_canonize=use_canonize,
            use_lattice=use_lattice,
            use_angle=use_angle,
        )

    # TODO: add if parallel ?
    pandarallel.initialize(progress_bar=False)
    graphs = df["cif"].parallel_apply(atoms_to_graph)
    # graphs = df["atoms"].apply(atoms_to_graph).values

    return graphs


def get_pyg_dataset(
    dataset=[],
    target="",
    neighbor_strategy="",
    atom_features="",
    use_canonize="",
    line_graph="",
    cutoff=8.0,
    max_neighbors=12,
    classification=False,
    # output_dir=".",
    # tmp_name="dataset",
    use_lattice=False,
    use_angle=False,
    # data_from="Jarvis",
    # use_save=False,
    mean_train=None,
    std_train=None,
    eval=False,
    # now=False,  # for test
):
    """Get pyg Dataset."""
    if isinstance(dataset, Path):
        df = pd.read_csv(dataset)
        vals = df[target].values
        print("data range", np.max(vals), np.min(vals))
        print("graphs not saved")
    else:
        assert mean_train is not None
        df = dataset
    graphs = load_pyg_graphs(
        df,
        neighbor_strategy=neighbor_strategy,
        use_canonize=use_canonize,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        use_lattice=use_lattice,
        use_angle=use_angle,
    )
    if mean_train is None:
        mean_train = np.mean(vals)
        std_train = np.std(vals)
        data = PygStructureDataset(
            df,
            graphs,
            target=target,
            atom_features=atom_features,
            line_graph=line_graph,
            classification=classification,
            neighbor_strategy=neighbor_strategy,
            mean_train=mean_train,
            std_train=std_train,
            eval=eval,
        )
    else:
        data = PygStructureDataset(
            df,
            graphs,
            target=target,
            atom_features=atom_features,
            line_graph=line_graph,
            classification=classification,
            neighbor_strategy=neighbor_strategy,
            mean_train=mean_train,
            std_train=std_train,
            eval=eval,
        )
    return data, mean_train, std_train
