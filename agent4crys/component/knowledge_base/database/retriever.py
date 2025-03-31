import os
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import faiss
from pymatgen.core.composition import Composition
from CBFV import composition


class ReasonRetriever:
    def __init__(self, elem_prop="onehot", mat_prop="formation_energy_per_atom"):
        self.mat_prop = mat_prop
        self.elem_prop = elem_prop
        vec_db, self.reason_db = self._read_db()
        # normalize vec_db
        vec_db = vec_db.to_numpy()
        norms = np.linalg.norm(vec_db, axis=1, keepdims=True)
        self.vec_db = vec_db / norms
        if self.elem_prop == "onehot":
            self.emb = 714
        elif self.elem_prop == "magpie":
            self.emb = 132
        else:
            raise ValueError("elem_prop must be onehot or magpie")
        self.index = faiss.IndexFlatIP(self.emb)
        self.index.add(self.vec_db)

    def _read_db(self):
        filename = f"comp_vec_{self.elem_prop}.csv"
        path = Path(os.path.dirname(__file__))
        vec_db = pd.read_csv(path / filename)
        with open(path / "responses_0_10000.jsonl", "r") as f:
            reason_db = [json.loads(line) for line in f]
        pairs = pd.read_csv(path / "pairs.csv")
        comp1_indexs = pairs["comp_1"].to_numpy()
        vec_db = vec_db.loc[comp1_indexs]
        return vec_db, reason_db

    def search(self, query, k=10):
        df_cbfv = pd.DataFrame(columns=["formula", "target"])
        comp = Composition(query)
        comp = comp.reduced_formula
        df_cbfv.loc[0] = [comp, 0.0]
        vec, _, _, _ = composition.generate_features(df_cbfv, self.elem_prop)
        vec = vec.to_numpy()
        vec = vec / np.linalg.norm(vec)

        _, indexs = self.index.search(vec, k)
        indexs = indexs[0]
        output = [self.reason_db[i] for i in indexs]
        return output

    def get_related_context(self, proposer, query, memory, k=5):
        search_results = self.search(query, k)
        items = self.rank_with_llm(proposer, memory, search_results)
        items = self.extract_outputs(items)
        formatted_output = ""
        for i, item in enumerate(items):
            item = search_results[item]
            formatted_output += f"""
* When the composition {item["comp_1"]} changes to the composition {item["comp_2"]}, the property value of the corresponding material changes from {item["prop_1"]:.3f} eV/atom to {item["prop_2"]:.3f} eV/atom.
 This is likely because of the following reasons: {item["response"]}
"""
        return formatted_output

    def rank_with_llm(self, proposer, memory, search_results, num_choices=3):
        system_prompt = proposer.general_system_prompt

        prompt = f"""
{proposer.target_prompt}
Your objective is to propose a new composition that achieve the desired property by exploring the materials space across the periodic table.
In the previous step, you suggested the composition {memory.memory["composition"][-1]}, and got the following feedback:
{memory.memory["feedback"][-1]}

To improve your suggestion, first review the information from the external knowledge base and extract the most insightful details.
Below is a list of descriptions, obtained from the external knowledge base based on your current suggestion, that illustrate how property values change with variations in material composition.
"""
        for i, item in enumerate(search_results):
            prompt += f"""
{i}. When the composition {item["comp_1"]} changes to the composition of {item["comp_2"]}, the property value of the corresponding material changes from {item["prop_1"]:.3f} eV/atom to {item["prop_2"]:.3f} eV/atom.
 This is likely because of the folowing reasons: {item["response"]}
"""

        prompt += f"""
From the list above, please select {num_choices} choices that you consider most insightful for achieving the task.
Your output must be a list in the following format:
[X, Y, Z]
Where X, Y, Z are the indices of the selected choices from 0 to {len(search_results)-1}.
"""
        response = proposer.generate(system_prompt, prompt)

        return response

    def extract_outputs(self, response):
        pattern = r"\[[^]]*\]"
        match = re.findall(pattern, response)
        assert match, "No match found"
        list_str = match[-1]
        items = json.loads(list_str)
        return items
