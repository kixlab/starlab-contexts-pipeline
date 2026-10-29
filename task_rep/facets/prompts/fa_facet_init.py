"""
Prompts for bootstrapping the iterative algorithm with common-sense contextual factors.

Unlike discovery (which resolves specific collisions), initialization asks the
LLM to propose N broad facets upfront given the tutorials.
"""

import logging
from typing import Any, Dict, List

from utils.llm import MessageContent
from utils import random_uid

from .fa_pydantic import ContextualFactorInitResult
from .formatters import (
    videos_to_message_content,
    build_item_id_map,
)

from facets.schema import Facet, FacetLabel
from knowledge_items.schema import KnowledgeItem

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You design user-facing metadata for tutorial videos about: {task}.

You propose contextual factors: dimensions of a learner's situation that determine which tutorial segments apply to them. A learner's situation consists of their goal, materials, equipment, and progress through the task.

Your audience is novice and intermediate learners of {task}. Factors become filter menus, timeline labels, and search options. Learners use them to find useful segments before watching, so every factor must be something a learner can state about their own situation without seeing the footage."""

# USER_PROMPT = """You are given a tutorial task domain. Identify exactly {num_facets} contextual factors that could be used to organize tutorial segments by differences in procedural context. The factors should capture important dimensions of variation in how tutorial content may apply, and should be distinct from each other.

# A contextual factor is an aspect of the procedural situation that helps characterize when a tutorial segment is applicable.

# For each factor:
# - provide a short title
# - provide a brief definition
# - provide a finite set of labels
# - provide brief segmentation guidance for when labels change

# Prefer factors that are interpretable, broadly useful, and likely to generalize across many tutorials in the domain."""

USER_PROMPT = """You are given tutorial segments from tutorials. Identify exactly {num_facets} contextual factors that can organize segments by learner situation. Propose the strongest, broadest factors first; once the obvious dimensions are covered, continue with progressively finer-grained learner-situation factors (specific materials, equipment choices, sub-goals, variations, problem conditions) until you reach {num_facets}. Never fill remaining slots with footage-derived or motor-description factors — go finer-grained within the learner's situation instead.

WHAT QUALIFIES AS A CONTEXTUAL FACTOR
Each vocabulary value must be a situation a learner could truthfully claim before watching. Test every value with this sentence:

    "Show me the segments for when ___."

- If a value can only be determined by observing how the instructor moves, holds, paces, presents, or what the camera shows, it describes the footage, not a learner's situation.
- A factor is useful only if learners in different situations need different segments. Before proposing it, identify two concrete learners whose situations differ on this factor and who would retrieve different segments.
- Values must use the words of {task}, not vocabulary from robotics, motor control, cinematography, or another discipline.
- If the factor and vocabulary would transfer unchanged to an unrelated manual task such as pottery, knitting, surgery, or furniture assembly, reject it.

VALIDITY REQUIREMENTS
- Distinct from other proposed factors.
- Grounded in the provided segments, with at least two values that would be used.
- Not based on narration, editing, camera framing, viewer visibility, visual occlusion, or segment presentation.

EXPRESSION REQUIREMENTS
- `title`: 1-3 everyday words; a concrete noun phrase a learner would recognize.
- `user_question`: the learner-facing question this factor answers, e.g. "What kind of roll are you making?" or "What tool do you have?"
- `vocabulary`: 3-8 values when possible. Labels are 1-3 common words and distinguishable by a first-time learner. One "other" or "not applicable" value is allowed.
- `definition` and value definitions: one plain sentence, roughly 15 words or fewer.
- `guidelines`: annotator-facing edge cases, tie-breakers, boundary conditions, and label-change triggers.

DEPTH RULE
Deliver exactly {num_facets} factors. When broad dimensions run out, do not stop and do not pad with footage-derived factors — instead subdivide the learner's situation more finely: specific ingredient or material choices, equipment and tool variants, preparation states, sub-goals, serving or output intentions, and common problem situations all qualify as long as each value still completes "Show me the segments for when ___".

EXAMPLE (different domain: pottery)
BAD: title="Prehension Patterns", user_question="How is the object grasped?", vocabulary=["precision pinch", "power grasp"].
GOOD: title="Clay Stage", user_question="How firm is your clay right now?", vocabulary=["wet", "leather-hard", "bone-dry", "fired"].

OUTPUT
For each proposed contextual factor, provide:
- `title`: short name (1-3 words)
- `title_plural`: plural form of the title
- `user_question`: a learner-facing question used to choose the factor value
- `scope`: one of "global", "mid-level", or "fine-grained"
- `definition`: brief description of what this factor captures
- `justification`: (1) why this is a valid contextual factor, and (2) why it is distinct from the previous contextual factors (if any)
- `guidelines`: list of concise instructions for labeling tutorial segments under this factor, including when the label should change
- `vocabulary`: finite list of mutually exclusive label values, each with `label` and `definition`

INPUTS
- Tutorial segments:
{tutorials}

REMINDER: every value must complete "Show me the segments for when ___"; use learner-situation vocabulary; return exactly {num_facets} factors, going finer-grained within the learner's situation rather than stopping early or padding with footage-derived factors.
"""

# USER_PROMPT = """You are given tutorial videos and information units extracted from them. Propose exactly {num_facets} broad `contextual factors` (facets) that would help differentiate when information units appear in the videos.

# CONTEXTUAL FACTOR DEFINITION
# A valid contextual factor is a variable describing the procedural situation at a given moment or span. It must be:

# - Domain-grounded: anchored in the task-world (e.g., procedural stage, material state, tool state, object configuration, environmental condition, operating mode, verification state, service context) — NOT in media, narration, editing, presentation, or social framing.
# - Consequential: changes what actions, constraints, interpretations, checks, or information apply.
# - Monolithic: captures one coherent semantic concept, not a bundle.
# - Piecewise-stable: remains constant over meaningful spans; changes at identifiable boundaries.
# - Target-independent: determined by the broader procedural context; removing the target item would not change the factor's value.

# Non-examples: video genre, tutorial format, presenter configuration, narration style, camera/editing style, entertainment framing, human skill level, subjective quality, labels restating target content.

# DESIDERATA
# Prefer factors that are:
# - easy to understand at a glance, even without deep domain familiarity
# - broadly useful across the tutorial
# - simple, intuitive, mid-level task-world distinctions over obscure or overly fine-grained ones

# GOAL
# Propose top-{num_facets} contextual factors that capture the most important axes of variation across the tutorials. Think about what meaningful dimensions a practitioner would naturally use to organize or contextualize the information in these tutorials.

# WHEN NOT TO PROPOSE FACTORS
# Do NOT force factors that are artificial or unreliable. Return an empty candidate list with a refusal reason if any of the following apply:
# - All target items occur in the same narrow region of the tutorial and share identical context — no factor can assign them different labels without reading the item content itself (violating target-independence).
# - The only distinguishing factors would be too fine-grained to generalize (e.g., "the exact product being discussed"), essentially re-encoding the item content as a label.
# - Proposed factors would require subjective or ambiguous judgment calls that different annotators would disagree on.
# It is better to return zero candidates than to propose a factor that does not reliably separate the information units.

# REQUIREMENTS
# - Each proposed factor must be distinct from the others.
# - For each factor, labeling must be non-overlapping, exhaustive, and single-label (exactly one label per moment in time).
# - Keep factor titles short (2-3 words)
# - Provide a finite, explicit label vocabulary for each factor
# - Prefer global or mid-level factors. Available context scopes are:
#     - Global: stable across a large section or the whole tutorial.
#     - Mid-level: persists for a meaningful span of the tutorial.
#     - Fine-grained (only if necessary): brief contexts used only if global + mid-level factors cannot suffice.

# PROCEDURE
# 1. Identify the {num_facets} most important contextual dimensions that vary across the tutorials.
#     - Brief validity note: one sentence confirming it satisfies the five criteria
#     - Provide for each factor:
#         - Context scope ("global", "mid-level", "fine-grained")
#         - Title
#         - Definition
#         - Segmentation guidelines (start/end rules; transcript cues for label change)
#         - Full vocabulary necessary to segment the tutorial according to the factor.

# OUTPUT
# Return the proposed contextual factors according to the schema.

# INPUTS
# - Example Tutorials:
# {tutorials}
# """


def form_facet_init_request(
    task: str,
    videos: Dict[str, List[KnowledgeItem]],
    num_facets: int,
    **kwargs,
) -> Dict[str, Any]:
    """Form request to initialize facets from tutorials."""
    all_items = [item for items in videos.values() for item in items]
    id_map = build_item_id_map(all_items)

    messages = [
        {
            "role": "system",
            "content": MessageContent.from_template(SYSTEM_PROMPT, task=task),
        },
        {
            "role": "user",
            "content": MessageContent.from_template(
                USER_PROMPT,
                task=task,
                num_facets=str(num_facets),
                tutorials=videos_to_message_content(videos, id_map),
            ),
        },
    ]
    return {
        "messages": messages,
        "response_format": ContextualFactorInitResult,
    }


def process_facet_init_response(
    response: Dict[str, Any],
    **kwargs,
) -> List[Facet]:
    """Process facet init response, returning typed Facet objects."""
    result = response.get("response", {})

    refusal_reason = result.get("refusal_reason", "")
    candidates = result.get("candidates", [])

    if not candidates:
        logger.info("Initialization proposal refused: %s", refusal_reason or "(no reason given)")
        return []

    logger.info("Initialization proposed %d facet(s).", len(candidates))

    facets = []
    for candidate in candidates:
        vocabulary = [
            FacetLabel(label=ld["label"].strip().lower(), definition=ld["definition"])
            for ld in candidate.get("vocabulary", [])
        ]
        facets.append(Facet(
            id=random_uid(),
            title=candidate["title"],
            title_plural=candidate["title_plural"],
            user_question=candidate.get("user_question", ""),
            scope=candidate["scope"],
            definition=candidate["definition"],
            justification=candidate["justification"],
            guidelines=candidate["guidelines"],
            vocabulary=vocabulary,
        ))
    return facets
