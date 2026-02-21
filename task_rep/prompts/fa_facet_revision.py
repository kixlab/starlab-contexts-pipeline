"""
Prompts for revising existing contextual factors (facets).

Revision can include:
- Adding new values to vocabularies
- Refining definitions and guidelines
- Splitting overly broad facets
- Merging redundant facets
"""

from typing import Any, Dict, List

from ..utils.llm import MessageContent
from .formatters import dict_mc_to_str

from .fa_pydantic import FacetRevisionResult

SYSTEM_PROMPT = """You are a helpful assistant who can analyze and refine contextual factors for the task: {task}."""

USER_PROMPT = """
You are given a list of contextual factors that enumerate possible task contexts (e.g., constraints, resources, approaches, goals, etc.) for a particular task. Your goal is to analyze the factors and propose new factors that help reduce the size of the list (i.e., total number of labels across all factors) either by "merging" a set of factors into one or "splitting" a factor into a set of smaller factors. Follow the procedure below.

### DEFINITIONS
- A contextual factor is a monolithic, `piecewise-stable`, `target-independent`, and `task-relevant` variable that spans over time (i.e., produces a temporal segmentation of the transcript).
    - Monolithic: a single, unified semantic concept that does not consist of multiple, distinct sub-concepts.
    - Piecewise-stable: stays constant for a meaningful span, then changes at clear boundaries.
    - Target-independent: the label is decided based on the comprehnsive context of the task at that moment/span, not the target piece itself. If you remove the specific target piece, the label should remain the same.
    - Task-relevant: reflects conditions that would plausibly change what task knowledge applies or does not apply.

### PROCEDURE
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
    existing_facets: Dict[str, MessageContent],
    **kwargs,
) -> Dict[str, Any]:
    """Form request to revise existing facets."""
    existing_facets_str = dict_mc_to_str(existing_facets)
    messages = [
        {
            "role": "system",
            "content": MessageContent.from_template(
                SYSTEM_PROMPT,
                task=task
            )
        },
        {
            "role": "user",
            "content": MessageContent.from_template(
                USER_PROMPT,
                existing_facets=existing_facets_str,
            )
        }
    ]
    return {
        "messages": messages,
        "response_format": FacetRevisionResult,
    }


def process_facet_revision_response(
    response: Dict[str, Any],
    **kwargs,
) -> List[Dict[str, Any]]:
    """Process response from facet revision."""
    result = response.get("response", {})
    candidates = result.get("candidates", [])
    return candidates
