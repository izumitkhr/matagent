import random
from pathlib import Path

import numpy as np
import torch
import pandas as pd
from pymatgen.core.structure import Structure
from tqdm import tqdm
import faiss

from agent4crys.component.proposer.util.prompt import (
    instruct_simple_output_format,
    GENERAL_SYSTEM_PROMPT,
)
from agent4crys.component.contriever.pretrain.util import load_model, load_dataloader


def get_initial_guess(args, proposer, prompt, additional_prompt="", device="cuda"):
    if args.initial_guess == "random":
        out_dict = get_init_from_random(args)
        return out_dict
    elif args.initial_guess == "llm":
        out_dict = get_init_from_llm(args, proposer, prompt, additional_prompt)
        return out_dict
    elif args.initial_guess == "retriever":
        out_dict = get_init_from_retriever(args, device)
        return out_dict
    else:
        raise ValueError(f"Unknown initial guess: {args.initial_guess}")


def get_init_from_random(args):
    # generate initial composition from training data
    path = Path(args.data_path)
    df = pd.read_csv(path)
    idx = random.randint(0, len(df))
    crystal = Structure.from_str(df.iloc[idx]["cif"], fmt="cif")
    comp = crystal.composition.reduced_formula
    out_dict = {"reflection": None, "reason": None, "composition": comp}
    return out_dict


def get_init_from_llm(args, proposer, target_prompt, additional_prompt):
    system_prompt = GENERAL_SYSTEM_PROMPT
    prompt = (
        target_prompt
        + " Could you suggest one possible material composition?"
        + instruct_simple_output_format()
        + additional_prompt
    )
    response = proposer.generate(system_prompt=system_prompt, prompt=prompt)
    out_dict = proposer.extract_outputs(response)
    return out_dict


def get_init_from_retriever(args, device="cuda", batch_size=1024):
    model, tokenizer = load_model()

    prompt = f"Below is a description of a material. The formation energy per atom is {args.target_value:.3f} eV/atom."
    embedding = tokenizer(prompt, truncation=True, max_length=128, padding="max_length")
    ids, mask = torch.tensor(embedding["input_ids"]), torch.tensor(
        embedding["attention_mask"]
    )
    with torch.no_grad():
        out, _ = model(
            ids.unsqueeze(0), ids.unsqueeze(0), mask.unsqueeze(0), mask.unsqueeze(0)
        )
    query = out.detach().numpy()
    query = query / np.linalg.norm(query)

    # read mp_20
    model = model.to(device)
    loader, tokenizer = load_dataloader(
        base_model="t5-base",
        data_path=args.data_path,
        batch_size=batch_size,
        attributes=[],
    )
    emb_all = []
    with torch.no_grad():
        for batch in tqdm(loader):
            input_ids, attention_mask, _, _ = batch
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            out, _ = model(input_ids, input_ids, attention_mask, attention_mask)
            emb_all.append(out.cpu().numpy())
    emb_all = np.concatenate(emb_all, axis=0)
    emb_all = emb_all / np.linalg.norm(emb_all, axis=1)[:, None]

    index = faiss.IndexFlatIP(emb_all.shape[1])
    index.add(emb_all)
    _, indexs = index.search(query, args.n_init)

    # get compositions
    path = Path(args.data_path)
    df = pd.read_csv(path)
    crystals = [Structure.from_str(df.iloc[i]["cif"], fmt="cif") for i in indexs[0]]
    comps = [crystal.composition.reduced_formula for crystal in crystals]
    outputs = [
        {"reflection": None, "reason": None, "composition": comp} for comp in comps
    ]
    return outputs
