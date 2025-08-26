import random

from helpers import random_uid, get_response_pydantic, extend_contents

from pydantic_models.experiment_1 import StepTaxonomySchema, StepTaxonomyUpdateSchema, LabeledInfoListSchema

STEP_REQUIREMENTS = """
    a) Procedural: Describe essential procedural objectives for progressing in the task.
    b) Atomic: Each step should be atomic and self-contained.
    c) Not Overlapping: Each step should not overlap with other steps.
    d) Not Vague: Each step should not be vague and mention concrete actions.
    e) Generalizable: Each step should be generalizable to other instances of the task.
"""

SYSTEM_PROMPT_INITIALIZE_STEP_TAXONOMY = """
You specialize in processing instructional materials for a procedural task `{task}`.
"""

USER_PROMPT_INITIALIZE_STEP_TAXONOMY = """
<instructions>
Define a set of `steps` that meet the following requirements:
{step_requirements}
</instructions>
"""

def initialize_step_taxonomy(task):
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_INITIALIZE_STEP_TAXONOMY.format(task=task),
        },
        {
            "role": "user",
            "content": USER_PROMPT_INITIALIZE_STEP_TAXONOMY.format( step_requirements=STEP_REQUIREMENTS),
        },
    ]
    
    response = get_response_pydantic(messages, StepTaxonomySchema)
    steps = response["steps"]
    return steps

SYSTEM_PROMPT_UPDATE_STEP_TAXONOMY = """
You are an expert in analyzing procedural instructional content for the task `{task}`.
"""

USER_PROMPT_UPDATE_STEP_TAXONOMY = """
You are provided with:
1. An existing step taxonomy: a list of procedural steps that are atomic, generalizable, and non-overlapping.
2. A tutorial: a list of steps specific to one instance of the task.

<instructions>
Your goal is to update the existing taxonomy to accommodate the tutorial's steps. Follow these guidelines:

1. **Match First**: Attempt to map each step from the new tutorial to existing steps in the taxonomy. Prefer reuse over creating new steps.
2. **Add If Necessary**: If a step from the new tutorial cannot be reasonably mapped to any existing step (due to new functionality/actions, tools, materials, outcomes, or methods), then add a **new step** to the taxonomy.
3. **Generalize Carefully**: If a new step is similar but not identical to an existing one, consider generalizing the existing step so both cases are covered — but only if it does not become vague or overly broad.
4. **Maintain Principles**: Ensure the updated taxonomy maintains these properties:
{step_requirements}
Be rigorous and thoughtful — the goal is to build a robust taxonomy that can evolve to represent all valid procedural variants of the task.
</instructions>
"""

def update_step_taxonomy(task, tutorial, taxonomy):
    existing_taxonomy_str = ""
    for i, step in enumerate(taxonomy):
        existing_taxonomy_str += f"{i+1}. {step['step']} ({step['description']})\n"

    tutorial_str = tutorial.replace("\n", " ")
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_UPDATE_STEP_TAXONOMY.format(task=task),
        },
        {
            "role": "user",
            "content": USER_PROMPT_UPDATE_STEP_TAXONOMY.format(step_requirements=STEP_REQUIREMENTS),
        },
        {
            "role": "user",
            "content": f"<taxonomy>\n{existing_taxonomy_str}\n</taxonomy>"
        },
        {
            "role": "user",
            "content": f"<tutorial>\n{tutorial_str}\n</tutorial>"
        },
    ]

    response = get_response_pydantic(messages, StepTaxonomyUpdateSchema)
    steps = response["steps"]
    return steps


SYSTEM_PROMPT_EXTRACT_STEPS = """
You are an expert in analyzing procedural instructional content for the task `{task}`.
"""

USER_PROMPT_EXTRACT_STEPS = """
You are provided with:
1. A step taxonomy: a list of procedural steps that are atomic, generalizable, and non-overlapping.
2. A tutorial: an instructional text about the specific instance of the task.

<instructions>
Your goal is to extract all the steps from the tutorial that are relevant to the task. 

Follow these guidelines:
a) **Match First**: Attempt to map each step from the tutorial to existing steps in the taxonomy. Prefer reuse over creating new steps. Describe the step based on the information provided in the tutorial.
b) **Add If Necessary**: If a step from the tutorial cannot be reasonably mapped to any existing step (due to differences in materials, tools, outcomes, or methods), then add a **new step**. Describe the step based on the information provided in the tutorial.

Make sure that all the extracted steps follow these requirements:
{step_requirements}
</instructions>
"""

def extract_steps_based_on_taxonomy(task, tutorial, taxonomy):
    taxonomy_str = ""
    for i, step in enumerate(taxonomy):
        taxonomy_str += f"{i+1}. {step['step']} ({step['description']})\n"

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_EXTRACT_STEPS.format(task=task),
        },
        {
            "role": "user",
            "content": USER_PROMPT_EXTRACT_STEPS.format(step_requirements=STEP_REQUIREMENTS),
        },
        {
           "role": "user",
            "content": f"<taxonomy>\n{taxonomy_str}\n</taxonomy>"
        },
        {
            "role": "user",
            "content": f"<tutorial>\n{tutorial}\n</tutorial>"
        },
    ]

    response = get_response_pydantic(messages, StepTaxonomySchema)
    steps = response["steps"]
    return steps


INFORMATION_TYPES = [
    {
        "type": "explanation",
        "definition": "Reasons why the instruction was performed or the consequences of the instruction.",
        "examples": [
            "Because every time we wear our contact lenses, makeup and even dirt particles [...] might harm our eyes directly.",
            "And these will overhang a little to help hide the gap."
        ]
    },
    {
        "type": "tip",
        "definition": "Additional instructions or information that makes the instructions easier, faster, or more efficient",
        "examples": [
            "I fnd that it’s easier to do just a couple of layers at a time instead of all four layers at a time.",
        ]
    },
    {
        "type": "warning",
        "definition": "Actions that should be avoided",
        "examples": [
            "I don’t know but I would say avoid using bleach if you can.",
        ]
    },
    {
        "type": "description", ### Subgoal, Status, Tool spec.
        "definition": "A description of the materials, the intermediate state of the objects, or the tools and equipment used.",
        "examples": [
            "These are awesome beans, creamy texture, slightly nutty loaded with favor.",
            "Something sticky and dirty all through the back seat.",
        ]
    },
]

SYSTEM_PROMPT_LABEL_INFORMATION = """
You are an expert in analyzing procedural instructional content for the task `{task}`.
"""

USER_PROMPT_LABEL_INFORMATION = """
You are provided with a tutorial for the task.

<instructions>
Your goal is to extract all the information relevant to the task from the tutorial and label them according to the taxonomy of information types.

Follow these guidelines:
a) **Extract Information**: Extract information in chunks (i.e., a segment of text that refers to a single subject (i.e., a tool, a material, a step, etc.) and belongs to a single type in the taxonomy).
b) **Label Information**: Label each chunk of information with the corresponding type in the taxonomy.
</instructions>

<taxonomy>
{taxonomy}
</taxonomy>
"""

def extract_information(task, tutorial, taxonomy):
    pass



SYSTEM_PROMPT_LABEL_INFORMATION = """
You are an expert in analyzing procedural instructional content for the task `{task}`.
"""

USER_PROMPT_LABEL_INFORMATION = """
You are provided with:
1. A step taxonomy: a list of prototypical procedural steps.
1-1. Each step has a list of examples: a list of information that is relevant to the step.
2. A tutorial: an instructional text about the specific instance of the task.

<instructions>
Your goal is to label each piece of information in the tutorial with the corresponding step in the taxonomy. If a piece of information does not belong to any step in the taxonomy but is relevant to the task, create a new step that covers this information. 

Follow these guidelines:
a) **Match First**: Attempt to match each piece of information in the tutorial to existing steps in the taxonomy. Prefer reuse over creating new steps.
b) **Add If Necessary**: If a piece of information from the tutorial cannot be reasonably mapped to any existing step (due to new functionality/actions, tools, materials, outcomes, or methods) but is relevant to the task, then add a **new step** to the taxonomy.

Make sure that the steps in the updated taxonomy follow these requirements:
{step_requirements}
</instructions>
"""

def label_information(task, tutorial, taxonomy):
    existing_taxonomy_str = ""
    for i, step in enumerate(taxonomy):
        existing_taxonomy_str += f"<step>\n"
        existing_taxonomy_str += f"{step['step']} - {step['description']}\n"
        if len(step["examples"]) > 0:
            existing_taxonomy_str += "<examples>\n"
            ### randomly sample 3 examples
            random.shuffle(step["examples"])
            examples = step["examples"][:3]
            for example in examples:
                existing_taxonomy_str += f"  - {example}\n"
            existing_taxonomy_str += "</examples>\n"
        existing_taxonomy_str += "</step>\n"

    tutorial_str = tutorial.replace("\n", " ")
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_LABEL_INFORMATION.format(task=task),
        },
        {
            "role": "user",
            "content": USER_PROMPT_LABEL_INFORMATION.format(step_requirements=STEP_REQUIREMENTS),
        },
        {
            "role": "user",
            "content": f"<taxonomy>\n{existing_taxonomy_str}\n</taxonomy>"
        },
        {
            "role": "user",
            "content": f"<tutorial>\n{tutorial_str}\n</tutorial>"   
        },
    ]

    response = get_response_pydantic(messages, LabeledInfoListSchema)
    infos = response["infos"]
    updated_taxonomy = response["updated_taxonomy"]
    return infos, updated_taxonomy
