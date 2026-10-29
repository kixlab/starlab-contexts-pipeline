import json
import os

DATASETS_PATH = "./static/datasets/"

CROSS_TASK_TASKS = [
    "Change a Tire",
    "Build Simple Floating Shelves",
    "Make French Toast",
    "Make Irish Coffee",
]

CUSTOM_TASKS = [
    ### Food and Entertaining
    "How to Make a Sushi Roll",
    "How to Make Caramel Apples",
    "How to Make a Milkshake Without Ice Cream",
    "How to Grill Steak",
    "How to Make Scrambled Eggs in a Microwave",

    ### Home and Garden
    "How to Grow Hydrangea from Cuttings",
    "How to Grow a Pumpkin",
    "How to Clean Bathroom Tile",
    "How to Polish Stainless Steel",
    "How to Clean a Glass Top Stove",
    "How to Get Rid of a Wasp's Nest",

    # Holidays and Traditions
    "How to Plant a Living Christmas Tree",

    # Sports and Fitness
    "How to Wrap Your Hands for Boxing",
    "How to Catch Trout",

    # Arts and Entertainment
    "How to Make a Paper Hat",
]

SUBGOAL_DESCRIPTION = """
Subgoal: Objective of a subsection.
Example: "Now for the intricate layer that will give me the final webbing look."
"""

INSTRUCTION_DESCRIPTION = """
Instruction: Actions that the instructor performs to complete the task.
Example: "We're going to pour that into our silicone baking cups."
"""

TOOL_DESCRIPTION = """
Tool: Introduction of the materials, ingredients, and equipment to be used.
Example: "I'm also going to use a pair of scissors, a glue stick, some fancy tape or some regular tape."
"""


METHOD_DESCRIPTION = SUBGOAL_DESCRIPTION + INSTRUCTION_DESCRIPTION + TOOL_DESCRIPTION

TIP_DESCRIPTION = """
Tip: Additional instructions or information that makes instructions easier, faster, or more efficient.
Example: "I find that it's easier to do just a couple of layers at a time instead of all four layers at a time."
"""

WARNING_DESCRIPTION = """
Warning: Actions that should be avoided.
Example: "I don't know but I would say avoid using bleach if you can."
"""

SUPPLEMENTARY_DESCRIPTION = TIP_DESCRIPTION + WARNING_DESCRIPTION

JUSTIFICATION_DESCRIPTION = """
Justification: Reasons why the instruction was performed.
Example: "Because every time we wear our contact lenses, makeup and even dirt particles [...] might harm our eyes directly."
"""

EFFECT_DESCRIPTION = """
Effect: Consequences of the instruction.
Example: "And these will overhang a little to help hide the gap."
"""
EXPLANATION_DESCRIPTION = JUSTIFICATION_DESCRIPTION + EFFECT_DESCRIPTION

STATUS_DESCRIPTION = """
Status: Descriptions of the current state of the target object.
Example: "Something sticky and dirty all through the back seat."
"""

CONTEXT_DESCRIPTION = """
Context: Descriptions of the method or the setting.
Example: "[...] The process of putting on a tip by hand [...] takes a lot of patience but it can be done if you're in a pinch."
"""
TOOL_SPECIFICATION_DESCRIPTION = """
Tool Specification: Descriptions of the tools and equipment.
Example: "These are awesome beans, creamy texture, slightly nutty loaded with flavor."
"""

DESCRIPTION_DESCRIPTION = STATUS_DESCRIPTION + CONTEXT_DESCRIPTION + TOOL_SPECIFICATION_DESCRIPTION

OUTCOME_DESCRIPTION = """
Outcome: Descriptions of the final results of the procedure.
Example: "And now we have a dinosaur taggy blanket that wrinkles, so a fun gift for any baby on your gift giving list."
"""

REFLECTION_DESCRIPTION = """
Reflection: Summary, evaluation, and suggestions for the future about the overall procedure.
Example: "However, I am still concerned about how safe rubbing alcohol actually is to use so maybe next time, I will give vodka a try."
"""

CONCLUSION_DESCRIPTION = OUTCOME_DESCRIPTION + REFLECTION_DESCRIPTION

IMPORTANT_TYPE_DESCRIPTIONS_COARSE = {
    "Method": METHOD_DESCRIPTION,
    "Supplementary": SUPPLEMENTARY_DESCRIPTION,
    "Explanation": EXPLANATION_DESCRIPTION,
    "Description": DESCRIPTION_DESCRIPTION,
    "Conclusion": CONCLUSION_DESCRIPTION,
}

IMPORTANT_TYPE_DESCRIPTIONS_FINE = {
    "Method - Subgoal": SUBGOAL_DESCRIPTION,
    "Method - Instruction": INSTRUCTION_DESCRIPTION,
    "Method - Tool": TOOL_DESCRIPTION,
    "Supplementary - Tip": TIP_DESCRIPTION,
    "Supplementary - Warning": WARNING_DESCRIPTION,
    "Explanation - Justification": JUSTIFICATION_DESCRIPTION,
    "Explanation - Effect": EFFECT_DESCRIPTION,
    "Description - Status": STATUS_DESCRIPTION,
    "Description - Context": CONTEXT_DESCRIPTION,
    "Description - Tool Specification": TOOL_SPECIFICATION_DESCRIPTION,
    "Conclusion - Outcome": OUTCOME_DESCRIPTION,
    "Conclusion - Reflection": REFLECTION_DESCRIPTION,
}

IMPORTANT_TYPES_COARSE = ["Method", "Supplementary", "Explanation", "Description"]

IMPORTANT_TYPES_FINE = [
    "Method - Subgoal", "Method - Instruction", "Method - Tool",
    "Supplementary - Tip", "Supplementary - Warning",
    "Explanation - Justification", "Explanation - Effect",
    "Description - Status", "Description - Context", "Description - Tool Specification",
    "Conclusion - Outcome", "Conclusion - Reflection",
]

def get_dataset(task):
    version = "framework_raw"
    dataset_filepath = f"{DATASETS_PATH}{task.replace(' ', '_').lower()}_{version}.json"
    if os.path.exists(dataset_filepath):
        with open(dataset_filepath) as f:
            dataset = json.load(f)
        return dataset
    raise ValueError(f"Dataset for {task} not found")