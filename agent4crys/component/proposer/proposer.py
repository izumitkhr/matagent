import ast
import re

from .util.prompt import (
    instruct_target,
    instruct_output_format,
    instruct_output_format_one,
)
from .util.prompt import SYSTEM_PROMPT, GENERAL_SYSTEM_PROMPT, SYSTEM_PROMPT_ONE
from .util.parse import remove_subscripts


class Proposer:
    def __init__(self, target_val, target_prompt, knowledge_base):
        self.target_val = target_val
        self.target_prompt = target_prompt
        self.system_prompt = SYSTEM_PROMPT
        self.system_prompt_one = SYSTEM_PROMPT_ONE
        self.general_system_prompt = GENERAL_SYSTEM_PROMPT
        self.kb_table, self.kb_reason = knowledge_base

    def propose_one(
        self, memory, prev_guess, feedback, file=None, additional_prompt=""
    ):
        prompt = self.target_prompt + instruct_target(prev_guess, feedback)
        prompt += instruct_output_format_one()
        prompt += additional_prompt
        next_guess = self.generate(self.system_prompt_one, prompt)
        next_guess = remove_subscripts(next_guess)
        next_guess = self.extract_outputs(next_guess)

        if file is not None:
            file.write("# Prompt:\n")
            file.write(prompt + "\n\n")
            file.write("# Next guess:\n")
            file.write(str(next_guess) + "\n\n")

        return next_guess

    def retrieve_and_propose_table(
        self,
        memory,
        prev_valid,
        prev_guess,
        feedback,
        file=None,
        chars=300,
        additional_prompt="",
    ):
        prompt = self.target_prompt + instruct_target(prev_guess, feedback)

        if prev_valid:
            related_atoms_dict = self.kb_table.get_all_related_atoms(
                memory.memory["composition"][-1]
            )
            prompt += """
In this step, you can examine elements from the same group or with similar chemical properties to explore potential substitutions or optimizations.
Below are the elements related to previously suggested composition.
"""
            for elem, related_atoms in related_atoms_dict.items():
                prompt += f"{elem}: {', '.join(related_atoms)}\n"
        else:
            pass

        prompt += instruct_output_format(chars)
        prompt += additional_prompt

        next_guess = self.propose(prompt)

        if file is not None:
            file.write("# Prompt:\n")
            file.write(prompt + "\n\n")
            file.write("# Next guess:\n")
            file.write(str(next_guess) + "\n\n")

        return next_guess

    def retrieve_and_propose_reason(
        self,
        memory,
        prev_valid,
        prev_guess,
        feedback,
        file=None,
        chars=300,
        additional_prompt="",
    ):
        prompt = self.target_prompt + instruct_target(prev_guess, feedback)

        if prev_valid:
            related_context = self.kb_reason.get_related_context(
                proposer=self,
                query=memory.memory["composition"][-1],
                memory=memory,
                k=5,
            )
            prompt += f"""
In this step, you refer to an external database to understand how materials properties change when substituting one material to another.
Below is the information on how property values change with composition for several materials, which might be useful.
{related_context}
"""
        else:
            pass

        prompt += instruct_output_format(chars)
        prompt += additional_prompt

        next_guess = self.propose(prompt)

        if file is not None:
            file.write("# Prompt:\n")
            file.write(prompt + "\n\n")
            file.write("# Next guess:\n")
            file.write(str(next_guess) + "\n\n")

        return next_guess

    def retrieve_and_propose_short_term(
        self,
        memory,
        prev_guess,
        feedback,
        prev_valid,
        file=None,
        chars=300,
        additional_prompt="",
    ):
        prompt = self.target_prompt + instruct_target(prev_guess, feedback)

        if prev_valid:
            related_context = memory.get_short_term_memory()
            prompt += f"""
In this step, you review the most recent composition proposal to identify trends, refine approaches and avoid redundant or ineffective suggestions.
Below are your short-term memories based on your prior trials that might be helpful.
{related_context}
"""
        else:
            pass

        prompt += instruct_output_format(chars)
        prompt += additional_prompt

        next_guess = self.propose(prompt)

        if file is not None:
            file.write("# Prompt:\n")
            file.write(prompt + "\n\n")
            file.write("# Next guess:\n")
            file.write(str(next_guess) + "\n\n")

        return next_guess

    def retrieve_and_propose_long_term(
        self,
        memory,
        prev_valid,
        prev_guess,
        feedback,
        file=None,
        chars=300,
        additional_prompt="",
    ):
        prompt = self.target_prompt + instruct_target(prev_guess, feedback)

        if prev_valid:
            related_context = memory.get_long_term_memory(target=self.target_val)
            prompt += f"""
In this step, you refer to your past successful trials and insights to guide the new composition proposal.
Below are your long-term memories based on your prior trials that might be helpful.
{related_context}
"""
        else:
            pass

        prompt += instruct_output_format(chars)
        prompt += additional_prompt

        next_guess = self.propose(prompt)

        if file is not None:
            file.write("# Prompt:\n")
            file.write(prompt + "\n\n")
            file.write("# Next guess:\n")
            file.write(str(next_guess) + "\n\n")

        return next_guess

    def extract_outputs(self, response):
        try:
            matches = re.findall(r"\{.*?\}", response, re.DOTALL)
            parsed_dict = ast.literal_eval(matches[-1])
            reflection = parsed_dict.get("reflection", None)
            reason = parsed_dict.get("reason", None)
            composition = parsed_dict.get("composition", None)
            out_dict = {
                "reflection": reflection,
                "reason": reason,
                "composition": composition,
            }
            return out_dict
        except:
            out_dict = {
                "reflection": None,
                "reason": None,
                "composition": None,
            }
            return out_dict
