"""
Prompts for revising existing contextual factors (facets).

Revision can include:
- Adding new values to vocabularies
- Refining definitions and guidelines
- Splitting overly broad facets
- Merging redundant facets
"""
import logging
from typing import Any, Dict, List

from utils.llm import MessageContent
from utils import random_uid

from .fa_pydantic import FacetRevisionResult
from .formatters import facets_to_message_content

from facets.schema import Facet, FacetLabel

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful assistant who can analyze and refine contextual factors for the task: {task}."""

USER_PROMPT = """
You are given a list of contextual factors that enumerate possible task contexts (e.g., constraints, resources, approaches, goals, etc.) for a particular task. Your goal is to analyze the factors and propose new factors that help reduce the size of the list (i.e., total number of labels across all factors) either by "merging" a set of factors into one or "splitting" a factor into a set of smaller factors. Follow the procedure below.

CONTEXTUAL FACTOR DEFINITION
A valid contextual factor is a variable describing the procedural situation at a given moment or span. It must be:

- Domain-grounded: anchored in the task-world (e.g., procedural stage, material state, tool state, object configuration, environmental condition, operating mode, verification state, service context) — NOT in media, narration, editing, presentation, or social framing.
- Consequential: changes what actions, constraints, interpretations, checks, or information apply.
- Monolithic: captures one coherent semantic concept, not a bundle.
- Piecewise-stable: remains constant over meaningful spans; changes at identifiable boundaries.
- Target-independent: determined by the broader procedural context; removing the target item would not change the factor's value.

Non-examples: video genre, tutorial format, presenter configuration, narration style, camera/editing style, entertainment framing, human skill level, subjective quality, labels restating target content.

DESIDERATA
Prefer factors that are:
- easy to understand at a glance, even without deep domain familiarity
- broadly useful across the tutorial
- simple, intuitive, mid-level task-world distinctions over obscure or overly fine-grained ones

GOAL
Propose contextual factors to compress the given set of factors. 
Follow the requirements and the procedure below.

WHEN NOT TO PROPOSE FACTORS
Do NOT force factors that are artificial or unreliable. Return an empty candidate list with a refusal reason if any of the following apply:
- All target items occur in the same narrow region of the tutorial and share identical context — no factor can assign them different labels without reading the item content itself (violating target-independence).
- The only distinguishing factors would be too fine-grained to generalize (e.g., "the exact product being discussed"), essentially re-encoding the item content as a label.
- Proposed factors would require subjective or ambiguous judgment calls that different annotators would disagree on.
It is better to return zero candidates than to propose a factor that does not reliably separate the targets.

PROCEDURE
1) Identify opportunities for list compression:
    - Check for a set of factors that could be merged into one monolithic factor.
    - Check for a an factor that could benefit from being split into a set of smaller, monolithic factors that contribute fewer labels in total.
2) For each opportunity identified, propose a new factor that captures the merged/split factors. When defining the factor, do not refer to the merged/split factors, but formulate its own definition that describes what factor of the task context it captures.
    - If there no opportunities identified, return an empty list.
    - For the "merging" opportunity, propose a new monolithic factor. Justify why merging is effective.
    - For the "splitting" opportunity, propose a set of new, monolithic factors. Justify why splitting is effective.
    - Justify why the proposed factor satisfies each part of the definition of a contextual factor.

List of existing factors:
{existing_facets}
"""


def form_facet_revision_request(
    task: str,
    facets: List[Facet],
    **kwargs,
) -> Dict[str, Any]:
    """Form request to revise existing facets."""
    messages = [
        {
            "role": "system",
            "content": MessageContent.from_template(SYSTEM_PROMPT, task=task),
        },
        {
            "role": "user",
            "content": MessageContent.from_template(
                USER_PROMPT,
                existing_facets=facets_to_message_content(facets),
            ),
        },
    ]
    return {
        "messages": messages,
        "response_format": FacetRevisionResult,
    }


def process_facet_revision_response(
    response: Dict[str, Any],
    **kwargs,
) -> List[Facet]:
    """Process facet revision response, returning typed Facet objects.

    IDs are self-generated (not taken from the LLM response).
    """
    result = response.get("response", {})
    candidates = result.get("candidates", [])
    changes = result.get("changes", "")
    logger.info("Revision changes: %s", changes)
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
