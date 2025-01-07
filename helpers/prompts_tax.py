from typing import Tuple

from helpers import get_response_pydantic, get_response_pydantic_with_message, extend_contents, random_uid

from pydantic_models.tax import ProceduralElementsSchema
from pydantic_models.tax import StepSchema, OutputSchema, InputSchema
from pydantic_models.tax import CleanStepDescription

INCLUDE_IMAGES = False

SYS_PROMPT_MASK_STEP = """
Given a step description, paraphrase it to create a concise version that excludes any specific details such as tools, materials, ingredients, or the outcomes. Focus on the actions and methods involved in the step.
"""

def mask_step(step):
    messages = [
        {
            "role": "system",
            "content": SYS_PROMPT_MASK_STEP,
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"Step Description: {step}\n"
            }]
        },
    ]
    response = get_response_pydantic(messages, CleanStepDescription)
    return response["description"]


SYS_PROMPT_EXTRACTION = """
You specialize in analyzing tutorials for procedural tasks. Given the subtitles and frames from the tutorial video about the task `{task}`, extract the steps with its inputs (i.e., tools, materials/ingredients) and the final outcomes (i.e., results).
"""

def extract_procedural_elements(contents, task):
    messages = [
        {   
            "role": "system",
            "content": SYS_PROMPT_EXTRACTION.format(task=task),
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"## Tutorial:\n"
            }] + extend_contents(contents, include_ids=False, include_images=INCLUDE_IMAGES),
        },
    ]

    response = get_response_pydantic(messages, ProceduralElementsSchema)
    steps = response["steps"]
    outputs = response["outputs"]

    input = []
    step = []
    output = []

    for step_obj in steps:
        step.append({
            "index": step_obj["index"],
            # "description": mask_step(step_obj["detailed_description"]),
            "description": mask_step(step_obj["title"]),
            "title": step_obj["title"],
            "detailed_description": step_obj["detailed_description"],
        })
        for input_obj in step_obj["inputs"]:
            input.append({
                # "description": input_obj["description"],
                "description": input_obj["title"],
                "title": input_obj["title"],
                # "step_index": step_obj["index"],
            })
            # if input_obj["source_step_index"] != -1:
            #     output.append({
            #         # "description": input_obj["description"],
            #         "description": input_obj["title"],
            #         "title": input_obj["title"],
            #         "step_index": input_obj["source_step_index"],
            #     })
    for output_obj in outputs:
        output.append({
            # "description": output_obj["description"],
            "description": output_obj["title"],
            "title": output_obj["title"],
            # "step_index": output_obj["step_index"],
        })
    return input, step, output

SYS_PROMPT_AGGREGATION = """
You specialize in analyzing tutorials for procedural tasks.
Given a list of {element}s extracted from different tutorials for the task `{task}`, aggregate them to create a single {element} that represents all of them. Provide a concise description of the aggregated {element} that captures the essence of all the {element}s.
"""

def aggregate_procedural_elements(contents, element, task):
    ObjectSchema = None
    if element == "step":
        ObjectSchema = StepSchema
    elif element == "output":
        ObjectSchema = OutputSchema
    elif element == "input":
        ObjectSchema = InputSchema
    else:
        raise ValueError(f"Invalid element: {element}")
    
    messages = [
        {   
            "role": "system",
            "content": SYS_PROMPT_AGGREGATION.format(task=task, element=element),
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"## {element.capitalize()}s:\n"
            }] + extend_contents(contents, include_ids=True, include_images=INCLUDE_IMAGES),
        },
    ]

    response = get_response_pydantic(messages, ObjectSchema)
    return response


