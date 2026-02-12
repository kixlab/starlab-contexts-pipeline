"""
Prompts for discovering new contextual factors (facets) to discriminate knowledge items.

Supports multiple discovery strategies:
- SINGLE_VIDEO: Find facets within a single video (intra-video)
- VIDEO_PAIRS: Find facets by discriminating between two videos (TODO)
- VIDEO_TRIPLETS: Find facets by discriminating among three videos (TODO)
"""

from enum import Enum
from typing import Any, Dict, List

from task_rep.utils.llm import MessageContent

from task_rep.prompts.fa_pydantic import FacetDiscoveryResult

from task_rep.prompts.formatters import dict_mc_to_str


SYSTEM_PROMPT = """You are a helpful assistant who can analyze and detect the contextual factors around the task: {task}."""

USER_PROMPT_SINGLE_VIDEO = """You are given tutorial videos and target information units extracted from them. Identify `contextual factors` (facets) that would help differentiate when these information units appear in the videos.

A contextual factor is a `piecewise-stable`, `target-independent`, and `task-relevant` variable:
- Piecewise-stable: stays constant for meaningful spans of the video, changes at clear boundaries
- Target-independent: the label is decided based on the comprehnsive context of the task at the moment/span of the information unit, not the information unit itself. If you remove the specific information unit, the label should remain the same.
- Task-relevant: reflects conditions that change what information is applicable or not applicable within the task

GOAL
Propose contextual factors so that target information units receive DIFFERENT signatures (ideally unique). You are also given a previous set of factors under which all targets share the same signature, so avoid proposing factors that are similar or overlapping with the previous set.
Follow the requirements and the procedure below.

REQUIREMENTS
- Each proposed factor must be distinct from the previous set of factors.
- For each factor, labeling must be non-overlapping, exhaustive, and single-label (exactly one label per moment in time).
- Keep factor titles short (2-3 words)
- Provide a finite, explicit label vocabulary for each factor
- Try to propose global or mid-level factors first. Available context scopes are:
    - Global: stable across a large section or the whole tutorial.
    - Mid-level: persists for a meaningful span of the tutorial.
    - Fine-grained (only if necessary): brief contexts used only if global + mid-level factors cannot resolve the collisions.

PROCEDURE
1. Propose contextual factors that improve discrimination among the target information units.
    - Briefly justify: (1) why it is a contextual factor and satisfies each part of the definition, (2) how it separates (at least some of the) colliding information units, (3) why it is not contained in the previous set.
    - Provide for each factor:
        - Context scope ("global", "mid-level", "fine-grained")
        - Title
        - Definition
        - Segmentation guidelines (start/end rules; transcript cues for label change)
        - Full vocabulary necessary to segment the tutorial according to the factor.
    - It is okay to not propose any factors if the information units are indistinguishable within the tutorials.
2. If there are factors, segment the tutorials according to each of them.
    - For each factor, produce a segmentation of the tutorials: every moment in the tutorials must be assigned exactly one label from the factor's vocabulary.
3. If there are factors, verify that the signatures of the target information units will be different under the new factors.

OUTPUT
Return the proposed contextual factors and segmentation of the tutorials according to each factor.

INPUTS
- Previous contextual factors:
{previous_facets}
- Signature of the target information units under the previous set of factors:
{signature}
- Target information units:
{target_items}
- Tutorials:
{tutorials}
"""


# ============= Form/Process Functions =============

def form_facet_discovery_request(
    task: str,
    tutorial_contents: Dict[str, MessageContent],
    target_items: Dict[str, MessageContent],
    existing_facets: Dict[str, MessageContent],
    target_signature: str,
    **kwargs,
) -> Dict[str, Any]:
    """Form request to discover facets from tutorials."""

    tutorial_contents_str = dict_mc_to_str(tutorial_contents)
    target_items_str = dict_mc_to_str(target_items)
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
                USER_PROMPT_SINGLE_VIDEO,
                previous_facets=existing_facets_str,
                tutorials=tutorial_contents_str,
                target_items=target_items_str,
                signature=target_signature,
            )
        }
    ]
    return {
        "messages": messages,
        "response_format": FacetDiscoveryResult,
    }


def process_facet_discovery_response(
    response: Dict[str, Any],
    target_items: Dict[str, MessageContent],
    **kwargs,
) -> List[Dict[str, Any]]:
    """Process response from facet discovery.

    Derives target_item_ids from dict keys of the same target_items used in the request.
    """
    target_item_ids = list(target_items.keys())

    result = response.get("response", {})
    candidates = result.get("candidates", [])

    # Build per-item annotation and signature tracking
    temp_annotations = {item_id: {} for item_id in target_item_ids}
    temp_signatures = {item_id: "" for item_id in target_item_ids}

    for idx, candidate in enumerate(candidates):
        c_id = f"C{idx+1}"
        for item_id in target_item_ids:
            temp_annotations[item_id][c_id] = "na"

        for segmentation_meta in candidate.get("segmentations", []):
            for item_annotation in segmentation_meta.get("segmentation", []):
                item_id = item_annotation.get("item_id")
                label = item_annotation.get("label")
                if item_id in temp_annotations:
                    temp_annotations[item_id][c_id] = label

        for item_id in target_item_ids:
            temp_signatures[item_id] += (
                f"<{c_id}:{temp_annotations[item_id][c_id]}>"
            )

    signature_values = list(temp_signatures.values())
    total = len(signature_values)
    unique = len(set(signature_values))
    if total > 0:
        print(
            f"Effectiveness ({len(candidates)}/{total}) "
            f"({unique}/{total}) = {unique / total:.3f}"
        )
    return candidates