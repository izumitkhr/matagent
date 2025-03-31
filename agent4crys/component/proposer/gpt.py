import os

import openai

from .proposer import Proposer
from .util.parse import remove_subscripts

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.environ.get("CHATGPT_API_KEY")


class OAProposer(Proposer):
    def __init__(self, target_val, target_prompt, knowledge_base, gpt_model="gpt-4o"):
        super().__init__(
            target_val=target_val,
            target_prompt=target_prompt,
            knowledge_base=knowledge_base,
        )
        self.gpt_model = gpt_model

    def generate(self, system_prompt, prompt):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        response = openai.chat.completions.create(
            model=self.gpt_model, messages=messages
        )
        return response.choices[0].message.content

    def propose(self, prompt):
        system_prompt = self.system_prompt

        response = self.generate(system_prompt, prompt)
        response = remove_subscripts(response)

        return self.extract_outputs(response)
