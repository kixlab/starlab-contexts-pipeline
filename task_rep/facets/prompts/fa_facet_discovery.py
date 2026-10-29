"""
Prompts for discovering new contextual factors (facets) to discriminate knowledge items.

Supports multiple discovery strategies:
- SINGLE_VIDEO: Find facets within a single video (intra-video)
- VIDEO_PAIRS: Find facets by discriminating between two videos (TODO)
- VIDEO_TRIPLETS: Find facets by discriminating among three videos (TODO)
"""

import logging
from typing import Any, Dict, List

from utils.llm import MessageContent
from utils import random_uid

from .fa_pydantic import FacetDiscoveryResult
from .formatters import (
    facets_to_message_content,
    target_items_to_message_content,
    videos_to_message_content,
    build_item_id_map,
    reverse_id_map,
)

from facets.schema import Facet, FacetLabel
from knowledge_items.schema import KnowledgeItem

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a helpful assistant who can analyze and detect the contextual factors around the task: {task}."""

USER_PROMPT_SINGLE_VIDEO = """You are given tutorial videos and target information units extracted from them. Identify `contextual factors` (facets) that would help differentiate when these information units appear in the videos.

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
Propose contextual factors so that target information units receive DIFFERENT signatures (ideally unique). You are also given a previous set of factors under which all targets share the same signature, so avoid proposing factors that are similar or overlapping with the previous set.
Follow the requirements and the procedure below.

WHEN NOT TO PROPOSE FACTORS
Do NOT force factors that are artificial or unreliable. Return an empty candidate list with a refusal reason if any of the following apply:
- All target items occur in the same narrow region of the tutorial and share identical context — no factor can assign them different labels without reading the item content itself (violating target-independence).
- The only distinguishing factors would be too fine-grained to generalize (e.g., "the exact product being discussed"), essentially re-encoding the item content as a label.
- Proposed factors would require subjective or ambiguous judgment calls that different annotators would disagree on.
It is better to return zero candidates than to propose a factor that does not reliably separate the targets.

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
    - If no valid factors can be proposed, return an empty candidate list with a refusal reason.
2. If there are factors, assign a label to each TARGET information unit under each proposed factor.
    - For each factor, provide a brief reason for each target assignment explaining why the label applies based on the surrounding context (not the item content itself).
    - For each factor, also provide 2-4 anchor items (non-target items from the tutorial) that demonstrate the factor applies broadly and that different labels from the vocabulary are actually used across the tutorial. Choose anchors that receive DIFFERENT labels from each other to show the factor's range.
3. If there are factors, verify that the signatures of the target information units will be different under the new factors.

OUTPUT
Return the proposed contextual factors, target assignments, and anchor examples according to the schema.

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


def form_facet_discovery_request(
    task: str,
    videos: Dict[str, List[KnowledgeItem]],
    target_items: List[KnowledgeItem],
    facets: List[Facet],
    target_signature: str,
    **kwargs,
) -> Dict[str, Any]:
    """Form request to discover facets from tutorials."""
    # Build a single id_map covering all video items so target_items and
    # tutorials sections use consistent temp IDs.
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
                USER_PROMPT_SINGLE_VIDEO,
                previous_facets=facets_to_message_content(facets),
                tutorials=videos_to_message_content(videos, id_map),
                target_items=target_items_to_message_content(target_items, id_map),
                signature=target_signature,
            ),
        },
    ]
    return {
        "messages": messages,
        "response_format": FacetDiscoveryResult,
    }


def process_facet_discovery_response(
    response: Dict[str, Any],
    videos: Dict[str, List[KnowledgeItem]],
    target_items: List[KnowledgeItem],
    **kwargs,
) -> List[Facet]:
    """Process facet discovery response, returning typed Facet objects.

    Remaps temp item IDs back to real IDs for signature tracking.
    IDs for returned Facets are self-generated (not taken from LLM response).
    """
    all_items = [item for items in videos.values() for item in items]
    id_map = build_item_id_map(all_items)
    rev_id_map = reverse_id_map(id_map)

    # temp IDs for the target items (used to track signatures)
    target_temp_ids = set[str]()
    for item in target_items:
        temp_id = id_map.get(item.id)
        if temp_id is None:
            logger.error("Target item %r not found in id_map; skipping.", item.id)
        else:
            target_temp_ids.add(temp_id)

    result = response.get("response", {})

    refusal_reason = result.get("refusal_reason")
    if refusal_reason:
        logger.info("Discovery refused: %s", refusal_reason)
        return []

    candidates = result.get("candidates", [])

    # Track signatures for console diagnostics (using temp IDs throughout)
    temp_annotations = {temp_id: {} for temp_id in target_temp_ids}
    temp_signatures = {temp_id: "" for temp_id in target_temp_ids}

    for idx, candidate in enumerate(candidates):
        c_id = f"C{idx + 1}"
        valid_labels = {ld["label"] for ld in candidate.get("segment_labels", [])}
        for temp_id in target_temp_ids:
            temp_annotations[temp_id][c_id] = "na"

        for assignment in candidate.get("target_assignments", []):
            temp_id = assignment.get("item_id")  # LLM returns temp ID
            label = assignment.get("label")
            if temp_id not in rev_id_map:
                logger.error(
                    "Discovery hallucinated item_id %r in target_assignments. Skipping.",
                    temp_id,
                )
                continue
            if label not in valid_labels:
                logger.error(
                    "Discovery hallucinated label %r for item %r. Skipping.",
                    label,
                    temp_id,
                )
                continue
            if temp_id in target_temp_ids:
                temp_annotations[temp_id][c_id] = label

        for temp_id in target_temp_ids:
            temp_signatures[temp_id] += f"<{c_id}:{temp_annotations[temp_id][c_id]}>"

    signature_values = list(temp_signatures.values())
    total = len(signature_values)
    unique = len(set(signature_values))
    if total > 0:
        logger.info(
            f"Effectiveness ({len(candidates)} facets for {total} units) "
            f"({unique}/{total}) = {unique / total:.3f}"
        )

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
