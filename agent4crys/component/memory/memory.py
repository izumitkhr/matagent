import numpy as np


class Memory:
    def __init__(
        self,
        max_short_term_size=3,
        use_short_term_memory=False,
        max_long_term_size=3,
        use_long_term_memory=False,
    ):
        self.memory_planning = {
            "reflection": [],
            "choice": [],
            "reason": [],
        }
        self.memory = {
            "reflection": [],
            "composition": [],
            "formula_units": [],
            "reason": [],
            "predicted_value": [],
            "feedback": [],
        }
        self.max_short_term_size = max_short_term_size
        self.max_long_term_size = max_long_term_size
        self.use_short_term_memory = use_short_term_memory
        self.use_long_term_memory = use_long_term_memory

    def store_plan(self, plan):
        for key in plan.keys():
            self.memory_planning[key].append(plan[key])

    def delete_plan(self):
        for key in self.memory_planning.keys():
            self.memory_planning[key].pop()

    def get_situation(self, max_situation_size=5, unit="eV/atom"):
        num_trials = len(self.memory_planning["choice"])

        if num_trials == 0:
            output = f"You have not yet attempted to propose a new composition and do not have enough experience.\n"
        else:
            output = f"You have made {num_trials} attempts to propose new compositions so far.\n"

        if len(self.memory_planning["choice"]) == 0:
            output += f"The property value of the current material is {self.memory['predicted_value'][-1]:.3f} {unit}."
        elif len(self.memory_planning["choice"]) > max_situation_size:
            output += f"Here are the changes in the physical properties based on your {max_situation_size} latest proposals:\n"
            for i in range(max_situation_size, 0, -1):
                output += f"* You chose {self.memory_planning['choice'][-i]}, and the property value changed from {self.memory['predicted_value'][-i-1]:.3f} {unit} to {self.memory['predicted_value'][-i]:.3f} {unit}\n"
        else:
            for i in range(len(self.memory_planning["choice"]), 0, -1):
                output += f"* You chose {self.memory_planning['choice'][-i]}, and the property value changed from {self.memory['predicted_value'][-i-1]:.3f} {unit} to {self.memory['predicted_value'][-i]:.3f} {unit}\n"
        return output

    def store(self, next_guess, feedback, predicted_value, formula_units=None):
        for key in self.memory.keys():
            if key in ["reflection", "composition", "reason"]:
                self.memory[key].append(next_guess[key])
            elif key == "feedback":
                self.memory[key].append(feedback)
            elif key == "predicted_value":
                self.memory[key].append(predicted_value)
            # elif key == "formula_units":
            #     self.memory[key].append(formula_units)

    def get_memory(self, target=None):
        if self.use_short_term_memory:
            return self.get_short_term_memory()
        elif self.use_long_term_memory:
            assert target is not None
            return self.get_long_term_memory(target)

    def get_short_term_memory(self, unit="eV/atom"):
        num_memory = len(self.memory["composition"])
        if num_memory > self.max_short_term_size:
            num_memory = self.max_short_term_size
        else:
            num_memory = num_memory - 1

        output = ""
        if num_memory == 0:
            output = "You have not yet attempted to propose a new composition and do not have enough experience."
        else:
            for i in range(num_memory, 0, -1):
                output += f"""
* From the composition {self.memory["composition"][-i-1]}, you suggested the composition {self.memory["composition"][-i]} based on the reason: {self.memory["reason"][-i]}
The property value changed from {self.memory["predicted_value"][-i-1]:.3f} {unit} to {self.memory["predicted_value"][-i]:.3f} {unit}.
"""
        return output

    def get_long_term_memory(
        self,
        target,
        unit="eV/atom",
    ):
        num_memory = len(self.memory["composition"])
        if num_memory > self.max_long_term_size:
            num_memory = self.max_long_term_size
        else:
            num_memory = num_memory - 1

        output = ""
        if num_memory == 0:
            output = "You have not yet attempted to propose a new composition and do not have enough experience."
        else:
            val_hist = self.memory["predicted_value"][1:]
            best_idx = np.argsort(np.abs(np.array(val_hist) - target))[:num_memory]
            for idx in best_idx:
                idx_real = idx + 1
                output += f"""
* From the composition {self.memory["composition"][idx_real-1]}, you suggested the composition {self.memory["composition"][idx_real]} based on the reason: {self.memory["reason"][idx_real]}
The property value changed from {self.memory["predicted_value"][idx_real-1]:.3f} {unit} to {self.memory["predicted_value"][idx_real]:.3f} {unit}.
"""
        return output
