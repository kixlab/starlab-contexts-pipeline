from helpers import random_uid, get_response_pydantic, extend_contents

from pydantic_models.exp_1 import AllInstructionsSchema, AllInstructionsHSchema

SYSTEM_PROMPT_EXTRACT_INSTRUCTIONS_H = """
You are an expert in extracting step-by-step procedural instructions for {task} from a given tutorial. Break down the tutorial into fine-grained, atomic instructions, ensuring that each step is self-contained 
and does not nest other instructions.

- Extract each fine-grained instruction separately - do not group or nest them under broader steps.
- Each instruction should be concise and self-contained, making it easy to follow.
- Details should only elaborate on their respective instruction and must not introduce new steps or unrelated information.
- If certain details do not directly belong to any specific instruction, categorize them under a separate "Misc" instruction.
- Preserve the original order of the tutorial to maintain logical flow.
"""

def extract_instructions_h(text, task):
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_EXTRACT_INSTRUCTIONS_H.format(task=task),
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"## Tutorial:\n{text}\n"
            }]
        },
    ]
    
    response = get_response_pydantic(messages, AllInstructionsHSchema)
    instructions = response["instructions"]
    return instructions

SYSTEM_PROMPT_EXTRACT_DETAILS = """
You are an expert in processing and extracting information for {task} from a given tutorial. You are given a tutorial and a set of instructions. Extract useful details from the tutorial that correspond to each instruction. If a detail does not directly belong to any specific instruction, categorize it under a separate "Misc" instruction.
"""

def extract_details(text, instructions, task):
    instructions_str = "\n".join(instructions)
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_EXTRACT_DETAILS.format(task=task),
        },
        {
            "role": "user",
            "content": f"## Tutorial:\n{text}\n"
        },
        {
            "role": "user",
            "content": f"## Instructions:\n{instructions_str}\n",
        },
    ]
    
    response = get_response_pydantic(messages, AllInstructionsHSchema)
    instructions = response["instructions"]
    return instructions


SYSTEM_PROMPT_EXTRACT_INSTRUCTIONS = """
You are an expert in extracting step-by-step procedural instructions for {task} from a given tutorial. Break down the tutorial into fine-grained, atomic instructions.

- Each instruction should be concise and self-contained, making it easy to follow.
- Preserve the original order of the tutorial to maintain logical flow.
"""

def extract_instructions(text, task):
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_EXTRACT_INSTRUCTIONS.format(task=task),
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"## Tutorial:\n{text}\n"
            }]
        },
    ]
    
    response = get_response_pydantic(messages, AllInstructionsSchema)
    instructions = response["instructions"]
    return instructions