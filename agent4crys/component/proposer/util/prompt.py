GENERAL_SYSTEM_PROMPT = """
You are an expert in materials science who suggests compositions of crystal structures."""

SYSTEM_PROMPT = """
You are an expert in materials science who suggests compositions of crystal structures.
"Composition" describes the types and proportions of elements in a material.

Your final answer must be given as a Python dictionary in the following format:
{"reflection": $REFLECTION, "reason": $REASON, "composition": $COMPOSITION}
"""

SYSTEM_PROMPT_ONE = """
You are an expert in materials science who suggests compositions of crystal structures.
"Composition" describes the types and proportions of elements in a material.

Your final answer must be given as a Python dictionary in the following format:
{"composition": $COMPOSITION}
"""


def instruct_target(prev_guess, feedback):
    prompt = f"""
Your objective is to propose a new material composition that achieves the desired property by exploring the materials space across the periodic table.
In the previous step, you suggested the composition {prev_guess["composition"]}, and got the following feedback:
{feedback}
"""
    return prompt


def instruct_output_format_one():
    prompt = """
Based on the feedback, propose a new material composition to better achieve the desired property.
If previous suggestions are not successful enough, you may need to consider other material systems.

Your final answer must be given as a Python dictionary in the following format:
{"composition": $COMPOSITION}
Here are some requirements:
$COMPOSITION should be composed only of element symbols and digits, and should not include decimal numbers or symbols such as '-', '', '.', '{', '}', etc.
"""
    return prompt


def instruct_output_format(chars=300):
    prompt = """
Based on the above information, write a reflection on your previous suggestion, explaining how successful it was, the reasons for its success or failure, and what will be needed to further achieve the target property.
Based on the reflection, propose a new material composition to better achieve the desired property, providing a reason to justify your choice.
If previous suggestions are not successful enough, you may need to consider other material systems.

Your final answer must be given as a Python dictionary in the following format:
{"reflection": $REFLECTION, "reason": $REASON, "composition": $COMPOSITION}
Here are some requirements:"""

    prompt += f"""
$REFLECTION should be a string of {chars} characters or less.
$REASON should be a string of {chars} characters or less."""
    prompt += """
$COMPOSITION should be composed only of element symbols and digits, and should not include decimal numbers or symbols such as '-', '', '.', '{', '}', etc.
"""
    return prompt


def instruct_simple_output_format():
    prompt = """
Your answer must be given as a Python dictionary in the following format:
{"composition": $COMPOSITION}
Here is the requirements:"""

    prompt += """
$COMPOSITION should be composed only of element symbols and digits, and should not include decimal numbers or symbols such as '-', '', '.', '{', '}', etc.
"""
    return prompt
