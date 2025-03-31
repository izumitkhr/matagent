import pandas as pd
import torch

from pymatgen.core.composition import Composition
from pymatgen.core.periodic_table import DummySpecie, DummySpecies

from agent4crys.evaluation.pretrain import load_evaluation_model, get_dataloader


def load_evaluator(args, device):
    return Evaluator(args, device)


class Evaluator:
    def __init__(self, args, device):
        self.eval_model, self.eval_model_cfg, self.std, self.mean = (
            load_evaluation_model()
        )
        self.device = device
        self.eval_model.to(self.device)

    def get_dataloader(self, gen_mat_df: pd.DataFrame):
        loader, prepare_batch = get_dataloader(
            gen_mat_df,
            cfg=self.eval_model_cfg,
            std_train=self.std,
            mean_train=self.mean,
        )
        return loader, prepare_batch

    def evaluate(self, gen_mat_df: pd.DataFrame):
        loader, prepare_batch = self.get_dataloader(gen_mat_df)
        with torch.no_grad():
            for batch in loader:
                inputs, _ = prepare_batch(batch, self.device)
                outputs = self.eval_model(inputs)
                outputs = outputs * self.std + self.mean
        outputs = outputs.cpu().numpy()
        outputs = pd.DataFrame({"cif": gen_mat_df["cif"], "predicted": outputs})
        return outputs

    def generate_feedback(self, val, unit="eV/atom"):
        prompt = f"The formation energy per atom of material generated from the composition was {val:.3f} {unit}."
        return prompt

    def feedback(self, gen_mat_df, unit="eV/atom"):
        predicted = gen_mat_df["predicted"]
        idx = predicted.idxmin()
        predicted_val = predicted[idx]
        prompt = self.generate_feedback(predicted_val, unit)
        # cif = gen_mat_df.iloc[idx]["cif"]
        # formula_units = gen_mat_df.iloc[idx]["formula_units"]
        return prompt, predicted_val

    def check_validity(self, prev_guess, max_natoms=34):
        comp = prev_guess["composition"]
        if comp is None:
            valid = False
            feedback = "Output was not in the expected format. Please provide output in a valid format."
            return valid, feedback
        if "." in comp:
            valid = False
            feedback = f"'.' in included in {prev_guess["composition"]}. Plased do not use desimal numbers in composition formula."
            return valid, feedback
        try:
            comp = Composition(comp)
            if comp.num_atoms > max_natoms:
                valid = False
                feedback = f"Number of atoms in {prev_guess["composition"]} is greater than the maximum allowed number of atoms ({max_natoms})."
                return valid, feedback
            for elem in comp.elements:
                if isinstance(elem, (DummySpecies, DummySpecie)):
                    valid = False
                    feedback = f"{elem} is not a valid element. Please use a valid element symbol."
                    return valid, feedback
            valid, feedback = True, None
            return valid, feedback
        except:
            valid = False
            feedback = f"{prev_guess["composition"]} is not a valid representation of composition. Please provide a valid composition formula."
            return valid, feedback
