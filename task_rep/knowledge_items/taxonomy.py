"""
Taxonomy for atomic knowledge items.

Based on the classification system from server/helpers/dataset.py

<type>
    <title> Greeting </title>
    <definition> Opening and closing remarks. </definition>
    <subtypes>
        <subtype>
            <title> Opening </title>
            <definition> Starting remarks and instructor/channel introductions. </definition>
            <example> Hey, what's up you guys, Chef [...] here. </example>
        </subtype>
        <subtype>
            <title> Closing </title>
            <definition> Parting remarks and wrap-up. </definition>
            <example> Stay tuned, we'll catch you all later. </example>
        </subtype>
    </subtypes>
</type>

<type>
    <title> Overview </title>
    <definition> Main purpose of the video and its descriptions. </definition>
    <subtypes>
        <subtype>
            <title> Goal </title>
            <definition> Main purpose of the video and its descriptions. </definition>
            <example> Today, I'll show you a special technique which is totally special and about image pressing. </example>
        </subtype>
        <subtype>
            <title> Motivation </title>
            <definition> Reasons or background information on why the video was created. </definition>
            <example> [...] Someone is making a very special valentine's day meal for another certain special someone. </example>
        </subtype>
        <subtype>
            <title> Briefing </title>
            <definition> Rundown of how the goal will be achieved. </definition>
            <example> I'm pretty sure that just taking a pencil and putting it over the front and then putting a bunch of rubber bands around the pencil [...] that's going to do it. </example>
        </subtype>
    </subtypes>
</type>

<type>
    <title> Method </title>
    <definition> Actions that the instructor performs to complete the task. </definition>
    <subtypes>
        <subtype>
            <title> Subgoal </title>
            <definition> Objective of a subsection. </definition>
            <example> Now for the intricate layer that will give me the final webbing look. </example>
        </subtype>
        <subtype>
            <title> Instruction </title>
            <definition> Actions that the instructor performs to complete the task. </definition>
            <example> We're going to pour that into our silicone baking cups. </example>
        </subtype>
        <subtype>
            <title> Tool </title>
            <definition> Introduction of the materials, ingredients, and equipment to be used. </definition>
            <example> I'm also going to use a pair of scissors, a glue stick, some fancy tape or some regular tape. </example>
        </subtype>
    </subtypes>
</type>

<type>
    <title> Supplementary </title>
    <definition> Additional instructions or information that makes instructions easier, faster, or more efficient. </definition>
    <subtypes>
        <subtype>
            <title> Tip </title>
            <definition> Additional instructions or information that makes instructions easier, faster, or more efficient. </definition>
            <example> I find that it's easier to do just a couple of layers at a time instead of all four layers at a time. </example>
        </subtype>

        <subtype>
            <title> Warning </title>
            <definition> Actions that should be avoided. </definition>
            <example> I don't know but I would say avoid using bleach if you can. </example>
        </subtype>
    </subtypes>
</type>

<type>
    <title> Explanation </title>
    <definition> Reasons why the instruction was performed. </definition>
    <subtypes>
        <subtype>
            <title> Justification </title>
            <definition> Reasons why the instruction was performed. </definition>
            <example> Because every time we wear our contact lenses, makeup and even dirt particles [...] might harm our eyes directly. </example>
        </subtype>
        <subtype>
            <title> Effect </title>
            <definition> Consequences of the instruction. </definition>
            <example> And these will overhang a little to help hide the gap. </example>
        </subtype>
    </subtypes>
</type>

<type>
    <title> Description </title>
    <definition> Descriptions of the current state of the target object. </definition>
    <subtypes>
        <subtype>
            <title> Status </title>
            <definition> Descriptions of the current state of the target object. </definition>
            <example> Something sticky and dirty all through the back seat. </example>
        </subtype>
        <subtype>
            <title> Context </title>
            <definition> Descriptions of the method or the setting. </definition>
            <example> [...] The process of putting on a tip by hand [...] takes a lot of patience but it can be done if you're in a pinch. </example>
        </subtype>
        <subtype>
            <title> Tool Specification </title>
            <definition> Descriptions of the tools and equipment. </definition>
            <example> These are awesome beans, creamy texture, slightly nutty loaded with flavor. </example>
        </subtype>
    </subtypes>
</type>

<type>
    <title> Conclusion </title>
    <definition> Descriptions of the final results of the procedure. </definition>
    <subtypes>
        <subtype> 
            <title> Outcome </title>
            <definition> Descriptions of the final results of the procedure. </definition>
            <example> And now we have a dinosaur taggy blanket that wrinkles, so a fun gift for any baby on your gift giving list. </example>
        </subtype>
        <subtype>
            <title> Reflection </title>
            <definition> Summary, evaluation, and suggestions for the future about the overall procedure. </definition>
            <example> However, I am still concerned about how safe rubbing alcohol actually is to use so maybe next time, I will give vodka a try. </example>
        </subtype>
    </subtypes>
</type>

<type>
    <title> Miscellaneous </title>
    <definition> Personal stories, jokes, user engagement, and advertisements. </definition>
    <subtypes>
        <subtype>
            <title> Side Note </title>
            <definition> Personal stories, jokes, user engagement, and advertisements. </definition>
            <example> Tristan is back from basketball - He made it on the team so it's pretty exciting. </example>
        </subtype>
        <subtype>
            <title> Self-promotion </title>
            <definition> Promotion of the instructor of the channel (i.e. likes, subscription, notification, or donations). </definition>
            <example> So if you like this video, please give it a thumbs up and remember to subscribe. </example>
        </subtype>
        <subtype>
            <title> Bridge </title>
            <definition> Meaningless phrases or expressions that connect different sections. </definition>
            <example> And we're going to go ahead and get started. </example>
        </subtype>
        <subtype>
            <title> Filler </title>
            <definition> Conventional filler words. </definition>
            <example> Whoops. </example>
        </subtype>
    </subtypes>
</type>
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List


@dataclass
class TaxonomyCategoryInfo:
    """Structured metadata for a single atomic type."""
    definition: str
    examples: List[str] = field(default_factory=list)
    """One or more illustrative examples. Can be empty."""


class AtomicCategory(str, Enum):
    """Coarse-grained categories for atomic information."""

    GREETING = "Greeting"
    OVERVIEW = "Overview"
    METHOD = "Method"
    SUPPLEMENTARY = "Supplementary"
    EXPLANATION = "Explanation"
    DESCRIPTION = "Description"
    CONCLUSION = "Conclusion"
    MISC = "Miscellaneous"


class AtomicType(str, Enum):
    """Fine-grained types for atomic information."""
    # Greeting
    GREETING_OPENING = "Greeting - Opening"
    GREETING_CLOSING = "Greeting - Closing"

    # Overview
    OVERVIEW_GOAL = "Overview - Goal"
    OVERVIEW_MOTIVATION = "Overview - Motivation"
    OVERVIEW_BRIEFING = "Overview - Briefing"

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

    # Miscellaneous
    MISC_SIDE_NOTE = "Miscellaneous - Side Note"
    MISC_SELF_PROMO = "Miscellaneous - Self-promotion"
    MISC_BRIDGE = "Miscellaneous - Bridge"
    MISC_FILLER = "Miscellaneous - Filler"


# Type descriptions for VLM classification
TYPE_DESCRIPTIONS: Dict[AtomicType, TaxonomyCategoryInfo] = {
    AtomicType.GREETING_OPENING: TaxonomyCategoryInfo(
        definition="Starting remarks and instructor/channel introductions.",
        examples=["Hey, what's up you guys, Chef [...] here."],
    ),
    AtomicType.GREETING_CLOSING: TaxonomyCategoryInfo(
        definition="Parting remarks and wrap-up.",
        examples=["Stay tuned, we'll catch you all later."],
    ),
    AtomicType.OVERVIEW_GOAL: TaxonomyCategoryInfo(
        definition="Main purpose of the video and its descriptions.",
        examples=["Today, I'll show you a special technique which is totally special and about image pressing."],
    ),
    AtomicType.OVERVIEW_MOTIVATION: TaxonomyCategoryInfo(
        definition="Reasons or background information on why the video was created.",
        examples=["[...] Someone is making a very special valentine's day meal for another certain special someone."],
    ),
    AtomicType.OVERVIEW_BRIEFING: TaxonomyCategoryInfo(
        definition="Rundown of how the goal will be achieved.",
        examples=["I'm pretty sure that just taking a pencil and putting it over the front and then putting a bunch of rubber bands around the pencil [...] that's going to do it."],
    ),
    AtomicType.METHOD_SUBGOAL: TaxonomyCategoryInfo(
        definition="Objective of a subsection.",
        examples=["Now for the intricate layer that will give me the final webbing look."],
    ),
    AtomicType.METHOD_INSTRUCTION: TaxonomyCategoryInfo(
        definition="Actions that the instructor performs to complete the task.",
        examples=["We're going to pour that into our silicone baking cups."],
    ),
    AtomicType.METHOD_TOOL: TaxonomyCategoryInfo(
        definition="Introduction of the materials, ingredients, and equipment to be used.",
        examples=["I'm also going to use a pair of scissors, a glue stick, some fancy tape or some regular tape."],
    ),
    AtomicType.SUPPLEMENTARY_TIP: TaxonomyCategoryInfo(
        definition="Additional instructions or information that makes instructions easier, faster, or more efficient.",
        examples=["I find that it's easier to do just a couple of layers at a time instead of all four layers at a time."],
    ),
    AtomicType.SUPPLEMENTARY_WARNING: TaxonomyCategoryInfo(
        definition="Actions that should be avoided.",
        examples=["I don't know but I would say avoid using bleach if you can."],
    ),
    AtomicType.EXPLANATION_JUSTIFICATION: TaxonomyCategoryInfo(
        definition="Reasons why the instruction was performed.",
        examples=["Because every time we wear our contact lenses, makeup and even dirt particles [...] might harm our eyes directly."],
    ),
    AtomicType.EXPLANATION_EFFECT: TaxonomyCategoryInfo(
        definition="Consequences of the instruction.",
        examples=["And these will overhang a little to help hide the gap."],
    ),
    AtomicType.DESCRIPTION_STATUS: TaxonomyCategoryInfo(
        definition="Descriptions of the current state of the target object.",
        examples=["Something sticky and dirty all through the back seat."],
    ),
    AtomicType.DESCRIPTION_CONTEXT: TaxonomyCategoryInfo(
        definition="Descriptions of the method or the setting.",
        examples=["[...] The process of putting on a tip by hand [...] takes a lot of patience but it can be done if you're in a pinch."],
    ),
    AtomicType.DESCRIPTION_TOOL_SPECIFICATION: TaxonomyCategoryInfo(
        definition="Descriptions of the tools and equipment.",
        examples=["These are awesome beans, creamy texture, slightly nutty loaded with flavor."],
    ),
    AtomicType.CONCLUSION_OUTCOME: TaxonomyCategoryInfo(
        definition="Descriptions of the final results of the procedure.",
        examples=["And now we have a dinosaur taggy blanket that wrinkles, so a fun gift for any baby on your gift giving list."],
    ),
    AtomicType.CONCLUSION_REFLECTION: TaxonomyCategoryInfo(
        definition="Summary, evaluation, and suggestions for the future about the overall procedure.",
        examples=["However, I am still concerned about how safe rubbing alcohol actually is to use so maybe next time, I will give vodka a try."],
    ),
    AtomicType.MISC_SIDE_NOTE: TaxonomyCategoryInfo(
        definition="Personal stories, jokes, user engagement, and advertisements.",
        examples=["Tristan is back from basketball - He made it on the team so it's pretty exciting."],
    ),
    AtomicType.MISC_SELF_PROMO: TaxonomyCategoryInfo(
        definition="Promotion of the instructor or the channel (i.e. likes, subscription, notification, or donations).",
        examples=["So if you like this video, please give it a thumbs up and remember to subscribe."],
    ),
    AtomicType.MISC_BRIDGE: TaxonomyCategoryInfo(
        definition="Meaningless phrases or expressions that connect different sections.",
        examples=["And we're going to go ahead and get started."],
    ),
    AtomicType.MISC_FILLER: TaxonomyCategoryInfo(
        definition="Conventional filler words.",
        examples=["Whoops."],
    ),
}


# Mapping from fine-grained to coarse categories
TYPE_TO_CATEGORY: Dict[AtomicType, AtomicCategory] = {
    # Greeting
    AtomicType.GREETING_OPENING: AtomicCategory.GREETING,
    AtomicType.GREETING_CLOSING: AtomicCategory.GREETING,
    # Overview
    AtomicType.OVERVIEW_GOAL: AtomicCategory.OVERVIEW,
    AtomicType.OVERVIEW_MOTIVATION: AtomicCategory.OVERVIEW,
    AtomicType.OVERVIEW_BRIEFING: AtomicCategory.OVERVIEW,
    # Method
    AtomicType.METHOD_SUBGOAL: AtomicCategory.METHOD,
    AtomicType.METHOD_INSTRUCTION: AtomicCategory.METHOD,
    AtomicType.METHOD_TOOL: AtomicCategory.METHOD,
    # Supplementary
    AtomicType.SUPPLEMENTARY_TIP: AtomicCategory.SUPPLEMENTARY,
    AtomicType.SUPPLEMENTARY_WARNING: AtomicCategory.SUPPLEMENTARY,
    # Explanation
    AtomicType.EXPLANATION_JUSTIFICATION: AtomicCategory.EXPLANATION,
    AtomicType.EXPLANATION_EFFECT: AtomicCategory.EXPLANATION,
    # Description
    AtomicType.DESCRIPTION_STATUS: AtomicCategory.DESCRIPTION,
    AtomicType.DESCRIPTION_CONTEXT: AtomicCategory.DESCRIPTION,
    AtomicType.DESCRIPTION_TOOL_SPECIFICATION: AtomicCategory.DESCRIPTION,
    # Conclusion
    AtomicType.CONCLUSION_OUTCOME: AtomicCategory.CONCLUSION,
    AtomicType.CONCLUSION_REFLECTION: AtomicCategory.CONCLUSION,
    # Miscellaneous
    AtomicType.MISC_SIDE_NOTE: AtomicCategory.MISC,
    AtomicType.MISC_SELF_PROMO: AtomicCategory.MISC,
    AtomicType.MISC_BRIDGE: AtomicCategory.MISC,
    AtomicType.MISC_FILLER: AtomicCategory.MISC,
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


# ---------------------------------------------------------------------------
# Atomic subtype focus presets
# ---------------------------------------------------------------------------

@dataclass
class AtomicSubtypeConfig:
    """Named preset of atomic subtypes for optimization focus filtering.

    ``subtypes=[]`` means no filtering — all atomic items count towards schema
    evaluation.  When a non-empty list is given, only items with those subtypes
    are used for schema evaluation (the "focused" knowledge base); proposal,
    vocabulary, and labeling still operate on the full knowledge base.
    """
    name: str
    subtypes: List[AtomicType]


ATOMIC_ALL = AtomicSubtypeConfig(
    name="all",
    subtypes=[],
)
"""No subtype filtering — all atomic items count towards schema evaluation."""

ATOMIC_SEMI_IMPORTANT = AtomicSubtypeConfig(
    name="semi_important",
    subtypes=[
        AtomicType.METHOD_SUBGOAL,
        AtomicType.METHOD_INSTRUCTION,
        AtomicType.METHOD_TOOL,
        AtomicType.SUPPLEMENTARY_TIP,
        AtomicType.SUPPLEMENTARY_WARNING,
        AtomicType.EXPLANATION_JUSTIFICATION,
        AtomicType.EXPLANATION_EFFECT,
        AtomicType.DESCRIPTION_STATUS,
        AtomicType.DESCRIPTION_CONTEXT,
        AtomicType.DESCRIPTION_TOOL_SPECIFICATION,
        AtomicType.CONCLUSION_OUTCOME,
        AtomicType.CONCLUSION_REFLECTION,
    ],
)
"""Semi-important 12 atomic subtypes explicitly enumerated."""

ATOMIC_MOST_IMPORTANT = AtomicSubtypeConfig(
    name="most_important",
    subtypes=[
        AtomicType.METHOD_INSTRUCTION,
        AtomicType.EXPLANATION_JUSTIFICATION,
        AtomicType.SUPPLEMENTARY_TIP,
        AtomicType.SUPPLEMENTARY_WARNING,
    ],
)
"""The 4 most task-critical atomic subtypes."""

ATOMIC_SUBTYPE_PRESETS: Dict[str, AtomicSubtypeConfig] = {
    p.name: p for p in [ATOMIC_ALL, ATOMIC_SEMI_IMPORTANT, ATOMIC_MOST_IMPORTANT]
}