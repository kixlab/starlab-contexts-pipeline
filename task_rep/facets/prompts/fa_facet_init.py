"""
Prompts for bootstrapping the iterative algorithm with common-sense contextual factors.

Unlike discovery (which resolves specific collisions), initialization asks the
LLM to propose N broad, task-relevant facets upfront given the tutorials.
"""

import logging
from typing import Any, Dict, List

from utils.llm import MessageContent
from utils import random_uid

from .fa_pydantic import FacetInitResult
from .formatters import (
    videos_to_message_content,
    build_item_id_map,
)

from facets.schema import Facet, FacetLabel
from knowledge_items.schema import KnowledgeItem

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a helpful assistant who can analyze and detect the contextual factors around the task: {task}."""

USER_PROMPT = """You are given tutorial videos and information units extracted from them. Propose exactly {num_facets} broad `contextual factors` (facets) that would help differentiate when information units appear in the videos.

A contextual factor is a domain-grounded, task-relevant, monolithic, piecewise-stable, and target-independent variable that describes the procedural situation of the task at a given moment or over a meaningful span.
    - Domain-grounded: the factor is anchored in the task-world and describes a real aspect of the procedure as it unfolds, such as procedural stage, material state, tool state, object configuration, environmental condition, or service context. It does not describe the media, editorial, or presentation format through which the procedure is shown.
    - Task-relevant: the factor changes what actions, constraints, interpretations, or information are applicable in that part of the task.
    - Monolithic: the factor captures one coherent semantic concept rather than combining multiple distinct concepts.
    - Piecewise-stable: the factor remains constant over meaningful temporal segments and changes at identifiable boundaries.
    - Target-independent: the factor is determined by the broader procedural context, not by the specific information unit being labeled; removing that unit would not change the factor value.
Non-examples of contextual factors include:
    - video genre or tutorial format
    - presenter configuration
    - narration style
    - camera or editing style
    - entertainment framing
    - human skill or expertise
    - subjective quality judgments

GOAL
Propose top-{num_facets} contextual factors that capture the most important axes of variation across the tutorials. Think about what meaningful dimensions a practitioner would naturally use to organize or contextualize the information in these tutorials.

WHEN NOT TO PROPOSE FACTORS
Do NOT force factors that are artificial or unreliable. Return an empty candidate list with a refusal reason if any of the following apply:
- All target items occur in the same narrow region of the tutorial and share identical context — no factor can assign them different labels without reading the item content itself (violating target-independence).
- The only distinguishing factors would be too fine-grained to generalize (e.g., "the exact product being discussed"), essentially re-encoding the item content as a label.
- Proposed factors would require subjective or ambiguous judgment calls that different annotators would disagree on.
It is better to return zero candidates than to propose a factor that does not reliably separate the information units.

REQUIREMENTS
- Each proposed factor must be distinct from the others.
- For each factor, labeling must be non-overlapping, exhaustive, and single-label (exactly one label per moment in time).
- Keep factor titles short (2-3 words)
- Provide a finite, explicit label vocabulary for each factor
- Prefer global or mid-level factors. Available context scopes are:
    - Global: stable across a large section or the whole tutorial.
    - Mid-level: persists for a meaningful span of the tutorial.
    - Fine-grained (only if necessary): brief contexts used only if global + mid-level factors cannot suffice.

PROCEDURE
1. Identify the {num_facets} most important contextual dimensions that vary across the tutorials.
    - Briefly justify why each is a contextual factor (domain-grounded, task-relevant, monolithic, piecewise-stable, target-independent, task-relevant).
    - Provide for each factor:
        - Context scope ("global", "mid-level", "fine-grained")
        - Title
        - Definition
        - Segmentation guidelines (start/end rules; transcript cues for label change)
        - Full vocabulary necessary to segment the tutorial according to the factor.

OUTPUT
Return the proposed contextual factors according to the schema.

INPUTS
- Example Tutorials:
{tutorials}
"""


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
                num_facets=str(num_facets),
                tutorials=videos_to_message_content(videos, id_map),
            ),
        },
    ]
    return {
        "messages": messages,
        "response_format": FacetInitResult,
    }


def process_facet_init_response(
    response: Dict[str, Any],
    **kwargs,
) -> List[Facet]:
    """Process facet init response, returning typed Facet objects."""
    result = response.get("response", {})

    candidates = result.get("candidates", [])
    if not candidates:
        logger.info("Initialization returned no facets.")
        return []

    logger.info("Initialization proposed %d facet(s).", len(candidates))

    facets = []
    for candidate in candidates:
        vocabulary = [
            FacetLabel(label=ld["label"].strip().lower(), definition=ld["definition"])
            for ld in candidate.get("segment_labels", [])
        ]
        facets.append(Facet(
            id=random_uid(),
            title=candidate["title"],
            title_plural=candidate["title_plural"],
            scope=candidate["scope"],
            definition=candidate["definition"],
            justification=candidate["justification"],
            guidelines=candidate["guidelines"],
            vocabulary=vocabulary,
        ))
    return facets
