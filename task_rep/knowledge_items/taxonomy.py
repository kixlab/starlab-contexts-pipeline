"""
Taxonomy for atomic knowledge items.

Based on the classification system from server/helpers/dataset.py
"""

from enum import Enum
from typing import Dict


class AtomicCategory(str, Enum):
    """Coarse-grained categories for atomic information."""

    METHOD = "Method"
    SUPPLEMENTARY = "Supplementary"
    EXPLANATION = "Explanation"
    DESCRIPTION = "Description"
    CONCLUSION = "Conclusion"


class AtomicType(str, Enum):
    """Fine-grained types for atomic information."""

    # Method
    METHOD_SUBGOAL = "Method - Subgoal"
    METHOD_INSTRUCTION = "Method - Instruction"
    METHOD_TOOL = "Method - Tool"

    # Supplementary
    SUPPLEMENTARY_TIP = "Supplementary - Tip"
    SUPPLEMENTARY_WARNING = "Supplementary - Warning"

    # Explanation
    EXPLANATION_JUSTIFICATION = "Explanation - Justification"
    EXPLANATION_EFFECT = "Explanation - Effect"

    # Description
    DESCRIPTION_STATUS = "Description - Status"
    DESCRIPTION_CONTEXT = "Description - Context"
    DESCRIPTION_TOOL_SPECIFICATION = "Description - Tool Specification"

    # Conclusion
    CONCLUSION_OUTCOME = "Conclusion - Outcome"
    CONCLUSION_REFLECTION = "Conclusion - Reflection"


# Type descriptions for VLM classification
TYPE_DESCRIPTIONS: Dict[AtomicType, str] = {
    AtomicType.METHOD_SUBGOAL: """Objective of a subsection.
Example: "Now for the intricate layer that will give me the final webbing look.""",

    AtomicType.METHOD_INSTRUCTION: """Actions that the instructor performs to complete the task.
Example: "We're going to pour that into our silicone baking cups.""",

    AtomicType.METHOD_TOOL: """Introduction of the materials, ingredients, and equipment to be used.
Example: "I'm also going to use a pair of scissors, a glue stick, some fancy tape or some regular tape.""",

    AtomicType.SUPPLEMENTARY_TIP: """Additional instructions or information that makes instructions easier, faster, or more efficient.
Example: "I find that it's easier to do just a couple of layers at a time instead of all four layers at a time.""",

    AtomicType.SUPPLEMENTARY_WARNING: """Actions that should be avoided.
Example: "I don't know but I would say avoid using bleach if you can.""",

    AtomicType.EXPLANATION_JUSTIFICATION: """Reasons why the instruction was performed.
Example: "Because every time we wear our contact lenses, makeup and even dirt particles [...] might harm our eyes directly.""",

    AtomicType.EXPLANATION_EFFECT: """Consequences of the instruction.
Example: "And these will overhang a little to help hide the gap.""",

    AtomicType.DESCRIPTION_STATUS: """Descriptions of the current state of the target object.
Example: "Something sticky and dirty all through the back seat.""",

    AtomicType.DESCRIPTION_CONTEXT: """Descriptions of the method or the setting.
Example: "[...] The process of putting on a tip by hand [...] takes a lot of patience but it can be done if you're in a pinch.""",

    AtomicType.DESCRIPTION_TOOL_SPECIFICATION: """Descriptions of the tools and equipment.
Example: "These are awesome beans, creamy texture, slightly nutty loaded with flavor.""",

    AtomicType.CONCLUSION_OUTCOME: """Descriptions of the final results of the procedure.
Example: "And now we have a dinosaur taggy blanket that wrinkles, so a fun gift for any baby on your gift giving list.""",

    AtomicType.CONCLUSION_REFLECTION: """Summary, evaluation, and suggestions for the future about the overall procedure.
Example: "However, I am still concerned about how safe rubbing alcohol actually is to use so maybe next time, I will give vodka a try.""",
}


# Mapping from fine-grained to coarse categories
TYPE_TO_CATEGORY: Dict[AtomicType, AtomicCategory] = {
    AtomicType.METHOD_SUBGOAL: AtomicCategory.METHOD,
    AtomicType.METHOD_INSTRUCTION: AtomicCategory.METHOD,
    AtomicType.METHOD_TOOL: AtomicCategory.METHOD,
    AtomicType.SUPPLEMENTARY_TIP: AtomicCategory.SUPPLEMENTARY,
    AtomicType.SUPPLEMENTARY_WARNING: AtomicCategory.SUPPLEMENTARY,
    AtomicType.EXPLANATION_JUSTIFICATION: AtomicCategory.EXPLANATION,
    AtomicType.EXPLANATION_EFFECT: AtomicCategory.EXPLANATION,
    AtomicType.DESCRIPTION_STATUS: AtomicCategory.DESCRIPTION,
    AtomicType.DESCRIPTION_CONTEXT: AtomicCategory.DESCRIPTION,
    AtomicType.DESCRIPTION_TOOL_SPECIFICATION: AtomicCategory.DESCRIPTION,
    AtomicType.CONCLUSION_OUTCOME: AtomicCategory.CONCLUSION,
    AtomicType.CONCLUSION_REFLECTION: AtomicCategory.CONCLUSION,
}


# Common filters for experiments
IMPORTANT_TYPES_COARSE = [
    AtomicCategory.METHOD,
    AtomicCategory.SUPPLEMENTARY,
    AtomicCategory.EXPLANATION,
    AtomicCategory.DESCRIPTION,
]

IMPORTANT_TYPES_FINE = [
    AtomicType.METHOD_INSTRUCTION,
    # Add others as needed
]


def get_category(atomic_type: AtomicType) -> AtomicCategory:
    """Get the coarse category for a fine-grained type."""
    return TYPE_TO_CATEGORY[atomic_type]


def get_types_in_category(category: AtomicCategory) -> list[AtomicType]:
    """Get all fine-grained types in a category."""
    return [t for t, c in TYPE_TO_CATEGORY.items() if c == category]
