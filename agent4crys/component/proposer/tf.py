import sys

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaTokenizer, MistralForCausalLM

from .proposer import Proposer


def retrieve_model(model_id):
    if "Qwen" in model_id or "Llama" in model_id:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map="auto",
        )
    elif "Mistral" in model_id:
        model = MistralForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map="auto",
        )
    else:
        raise NotImplementedError(f"Model {model_id} not supported")
    return model


def retrieve_tokenizer(model_id):
    if "Qwen" in model_id or "Llama" in model_id:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    elif "Mistral" in model_id:
        tokenizer = LlamaTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


class TFProposer(Proposer):
    def __init__(
        self,
        target_val,
        target_prompt,
        knowledge_base,
        model_id,
        max_new_tokens=2048,
    ):
        super().__init__(target_val, target_prompt, knowledge_base=knowledge_base)
        self.tokenizer = retrieve_tokenizer(model_id)
        self.model = retrieve_model(model_id)
        self.max_new_tokens = max_new_tokens

    def generate(self, system_prompt, prompt):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        outputs = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :])
        return outputs
