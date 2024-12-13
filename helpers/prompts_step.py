from typing import Tuple

from helpers import get_response_pydantic, get_response_pydantic_with_message, extend_contents, random_uid

from pydantic_models.step import StepsSchema, TimedStepsSchema, StepSchema

SYS_PROMPT_INITIAL_STEPS = """
You specialize in creating novice-friendly tutorials for procedural tasks.

Define a set of `step` that can guide novices through the task `{task}`. Ensure each `step` meets the following requirements:
    a) Goal-Oriented: Represents an overarching objective or subtask that structures the process into meaningful phases.
    b) Procedural: Describes a procedural objective that is essential for progressing in the task.
    c) Generalizable: Accommodates variations in materials, tools, methods, and outcomes.
"""

def get_initial_steps(task):
    messages = [
        {
            "role": "system",
            "content": SYS_PROMPT_INITIAL_STEPS.format(task=task),
        },
    ]
    
    response = get_response_pydantic(messages, StepsSchema)
    steps = response["steps"]
    return steps

SYS_PROMPT_STEPS = """
You specialize in analyzing tutorials for procedural tasks.

Given a narration of the tutorial and a taxonomy of steps for the task `{task}`, annotate the steps in the narration with the corresponding steps from the taxonomy. If a step in the narration does not match any step in the taxonomy, create a new step.
All steps in the narration must meet the following requirements:
    a) Goal-Oriented: Represents an overarching objective or subtask that structures the process into meaningful phases.
    b) Procedural: Describes a procedural objective that is essential for progressing in the task.
    c) Generalizable: Accommodates variations in materials, tools, methods, and outcomes.
"""


def get_steps(contents, steps, task):
    messages = [
        {   
            "role": "system",
            "content": SYS_PROMPT_STEPS.format(task=task),
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"## Steps Taxonomy:\n"
            }] + extend_contents(steps, include_ids=False),
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"## Narration:\n"
            }] + extend_contents(contents, include_ids=True),
        },
    ]

    response = get_response_pydantic(messages, TimedStepsSchema)
    steps = response["steps"]
    steps = sorted(steps, key=lambda x: x["start_index"])
    return steps

SYS_PROMPT_AGGREGATE_STEPS = """
You specialize in analyzing tutorials for procedural tasks.

Given a set of steps for the task `{task}`, transform the steps into a single, coherent step that represents the entire set. Ensure the aggregated step meets the following requirements:
    a) Goal-Oriented: Represents an overarching objective or subtask that structures the process into meaningful phases.
    b) Procedural: Describes a procedural objective that is essential for progressing in the task.
    c) Generalizable: Accommodates variations in materials, tools, methods, and outcomes.
    d) Comprehensive: Encompasses all the steps in the set.
"""


def aggregate_steps(steps, task):
    messages = [
        {   
            "role": "system",
            "content": SYS_PROMPT_AGGREGATE_STEPS.format(task=task),
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"## Steps to Aggregate:\n"
            }] + extend_contents(steps, include_ids=False),
        },
    ]
    
    response = get_response_pydantic(messages, StepSchema)
    agg_step = response
    return agg_step