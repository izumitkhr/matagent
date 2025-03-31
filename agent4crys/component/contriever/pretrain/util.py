import os

from pathlib import Path
import torch
from transformers import T5Tokenizer
from torch.utils.data import DataLoader

from agent4crys.component.contriever.util.model import SimCLR
from agent4crys.component.contriever.util.data import MyDataset


def load_model():
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    model_path = current_dir / "best_model.pth"
    tokenizer = torch.load(model_path)["tokenizer"]
    model = SimCLR(tokenizer)
    model.load_state_dict(torch.load(model_path)["model_state_dict"])
    return model, tokenizer


def load_dataloader(base_model, data_path, batch_size, attributes=[]):
    tokenzier = T5Tokenizer.from_pretrained(base_model)

    dataset = MyDataset(data_path, tokenzier, attributes)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return dataloader, dataset.tokenizer
