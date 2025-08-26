from helpers import random_uid, get_response_pydantic, extend_contents

from pydantic_models.exp_0 import SubgoalsSchema, SubgoalSchema

SYSTEM_PROMPT_DEFINE_INITIAL_SUBGOALS = """
You specialize in tutorials for a procedural task `{task}`.
Define a set of `subgoals` that meet the following requirements:
    a) Goal-Oriented: Represent overarching objectives or subtasks that structure the process into meaningful phases.
    b) Procedural: Describe essential procedural objectives for progressing in the task.
    c) Generalizable: Accommodates various methods and approaches for the task.
"""

def define_initial_subgoals(task):
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_DEFINE_INITIAL_SUBGOALS.format(task=task),
        },
    ]
    
    response = get_response_pydantic(messages, SubgoalsSchema)
    subgoals = response["subgoals"]
    return subgoals

SYSTEM_PROMPT_EXTRACT_SUBGOALS = """
You specialize in tutorials for a procedural task `{task}`.
Given subtitles of a tutorial about the task, extract a set of `subgoals` that meet the following requirements:
    a) Present: Based on the subtitles, the subgoals should be present in the tutorial.
    b) Goal-Oriented: Represent overarching objectives or subtasks that structure the process into meaningful phases.
    c) Procedural: Describe essential procedural objectives for progressing in the task.
    d) Generalizable: Accommodates various methods and approaches for the task.

Here is an example of potentially expected subgoals:
{example_subgoals}
"""


def extract_subgoals(contents, example_subgoals, task):
    messages = [
        {   "role": "system",
            "content": SYSTEM_PROMPT_EXTRACT_SUBGOALS.format(task=task, example_subgoals=example_subgoals),
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"# Subtitles:\n"
            }] + extend_contents(contents, include_ids=True),
        },
    ]

    response = get_response_pydantic(messages, SubgoalsSchema)
    subgoals = response["subgoals"]
    return subgoals

SYSTEM_PROMPT_AGGREGATE_SUBGOALS = """
You specialize in tutorials for a procedural task `{task}`.
Given `subgoals` extracted from different tutorials, genereate a single comprehensive `subgoal` that generalize all the given subgoals.
"""

def aggregate_subgoals(subgoals, task):
    subgoals_str = "\n".join(subgoals)
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_AGGREGATE_SUBGOALS.format(task=task)},
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"## Subgoals:\n{subgoals_str}\n"
            }]
        },
    ]

    response = get_response_pydantic(messages, SubgoalSchema)

    subgoal = response
    return subgoal