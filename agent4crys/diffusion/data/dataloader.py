from torch_geometric.loader import DataLoader


def get_dataloader(splits, batch_size):
    dataloaders = {}
    for split, dataset in splits.items():
        if split == "train":
            dataloaders[split] = DataLoader(
                dataset, shuffle=True, batch_size=batch_size.train
            )
        elif split == "val":
            dataloaders[split] = DataLoader(
                dataset, shuffle=False, batch_size=batch_size.val
            )
        elif split == "test":
            dataloaders[split] = DataLoader(
                dataset, shuffle=False, batch_size=batch_size.test
            )
        else:
            raise ValueError(f"Unknown key: {split}")
    return dataloaders
