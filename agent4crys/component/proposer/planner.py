import ast
import re

from .proposer import Proposer
from .util.prompt import GENERAL_SYSTEM_PROMPT


class Planner:
    def __init__(self, proposer):
        super().__init__()
        self.proposer = proposer

        self.choices = {
            "id": [1, 2, 3, 4],
            "title": [
                "Refer to short-term memory",
                "Refer to long-term memory",
                "Refer to periodic table",
                "Refer to an external database",
            ],
            "description": [
                "Review the most recent composition proposals to identify trends, refine approaches and avoid redundant or ineffective suggestions.",
                "Refer to your past successful trials and insights to guide the new composition proposal.",
                "Examine elements from the same group or with similar chemical properties to explore potential substitutions or optimizations.",
                "Examine the external materials knowledgebase to understand how materials properties change when substituting one material to another.",
            ],
            "advantage": [
                "Helps maintain continuity and prevents unnecessary repetition or divergence from recent progress.",
                "Leverages accumulated knowledge and experience, increasing the likelihood of identifying promising compositions.",
                "Provides a systematic way to predict how modifying the composition might impact properties based on elemental trends.",
                "Enables data-driven predictions by learning from known material transformations and their effects on properties.",
            ],
        }

    def generate(self, system_prompt, prompt):
        output = self.proposer.generate(system_prompt, prompt)
        return output

    def extract_outputs(self, response):
        return self.proposer.extract_outputs(response)

    def plan_and_execute(
        self,
        memory,
        prev_guess,
        feedback,
        prev_valid,
        chars=300,
        file=None,
        additional_prompt="",
    ):
        plan_dict = self.plan(memory=memory, chars=chars, file=file)
        memory.store_plan(plan_dict)

        next_guess = self.execute(
            plan_dict=plan_dict,
            memory=memory,
            prev_guess=prev_guess,
            feedback=feedback,
            prev_valid=prev_valid,
            chars=chars,
            file=file,
            additional_prompt=additional_prompt,
        )
        return next_guess

    def _format_choices(self) -> str:
        return "\n".join(
            [
                f"{self.choices["id"][i]}. {self.choices["title"][i]} ({self.choices["description"][i]})"
                for i in range(len(self.choices["id"]))
            ]
        )

    def execute(
        self,
        plan_dict,
        memory,
        prev_guess,
        feedback,
        prev_valid,
        chars=300,
        file=None,
        additional_prompt="",
    ):
        choice = plan_dict["choice"]
        if choice == 1:  # short term memory
            next_guess = self.proposer.retrieve_and_propose_short_term(
                memory=memory,
                prev_guess=prev_guess,
                feedback=feedback,
                prev_valid=prev_valid,
                file=file,
                chars=chars,
                additional_prompt=additional_prompt,
            )
        elif choice == 2:  # long term memory
            next_guess = self.proposer.retrieve_and_propose_long_term(
                memory=memory,
                prev_valid=prev_valid,
                prev_guess=prev_guess,
                feedback=feedback,
                file=file,
                chars=chars,
                additional_prompt=additional_prompt,
            )
        elif choice == 3:  # periodic table
            next_guess = self.proposer.retrieve_and_propose_table(
                memory=memory,
                prev_valid=prev_valid,
                prev_guess=prev_guess,
                feedback=feedback,
                file=file,
                chars=chars,
                additional_prompt=additional_prompt,
            )
        elif choice == 4:  # external database
            next_guess = self.proposer.retrieve_and_propose_reason(
                memory=memory,
                prev_valid=prev_valid,
                prev_guess=prev_guess,
                feedback=feedback,
                file=file,
                chars=chars,
                additional_prompt=additional_prompt,
            )
        else:
            raise ValueError(f"Invalid choice: {choice}")
        return next_guess

    def _parse_response(self, response):
        matches = re.findall(r"\{.*?\}", response, re.DOTALL)
        try:
            parsed_dict = ast.literal_eval(matches[-1])
        except:
            parsed_dict = self._reformat_response(response)
        reflection = parsed_dict.get("reflection", None)
        reason = parsed_dict.get("reason", None)
        choice = parsed_dict.get("choice", None)
        out_dict = {
            "reflection": reflection,
            "reason": reason,
            "choice": choice,
        }
        return out_dict

    def plan(self, memory, chars=500, file=None):
        choices = self._format_choices()
        situation = memory.get_situation()

        prompt = f"""
{self.proposer.target_prompt}
Your objective is to propose a new material composition that achieves the desired property by exploring the material space across the periodic table.
To do so, you first need to assess the current situation and determine the optimal plan for your next step.
Below, you will find a list of possible strategies. After reflecting on the current state, evaluate each option by considering its advantages and risks. Then, select the most appropriate strategy and justify your choice.

Below are the choices you can make:
{choices}

Below is the current situation:
{situation}
"""
        prompt += """
Based on the above information, please reflect on the current state and make a choice with a reason.
Your final answer must be given as a Python dictionary in the following format:
{"reflection": $REFLECTION, "reason": $REASON, "choice": $CHOICE}"""

        prompt += f"""
Here are some requirements:
$REFLECTION should be a string of {chars} characters or less.
$REASON should be a string of {chars} characters or less.
$CHOICE should be an integer between 1 and {len(self.choices["id"])}.
"""
        response = self.proposer.generate(GENERAL_SYSTEM_PROMPT, prompt)
        response = self._parse_response(response)

        if file is not None:
            file.write("# Prompt:\n")
            file.write(prompt + "\n")
            file.write("# Response:\n")
            file.write(str(response) + "\n\n")

        return response

    def _reformat_response(self, response):
        system_prompt = (
            "You format the output of the response to match the expected format."
        )
        prompt = f"""
Please reformat the response to match the expected format.

Original response:
{response}"""

        prompt += """
Expected format:
{"reflection": $REFLECTION, "reason": $REASON, "choice": $CHOICE}
Here, $REFLECTION is a string enclosed in double quotes, $REASON is a string enclosed in double quotes, and $CHOICE is an integer.
"""
        response = self.proposer.generate(system_prompt, prompt)

        matches = re.findall(r"\{.*?\}", response, re.DOTALL)
        parsed_dict = ast.literal_eval(matches[-1])
        return parsed_dict
