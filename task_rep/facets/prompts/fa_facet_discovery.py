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

USER_PROMPT = """You are given a set of tutorial segments, a subset of target tutorial segments (i.e., targets), and previously identified contextual factors. Your task is to identify new contextual factors that can help distinguish the targets. The new factors should be different from the previous contextual factors and from each other.
A contextual factor is an aspect of the procedural situation that helps characterize when a tutorial segment is applicable.
For each factor:
- provide a short title
- provide a brief definition
- explain why it helps distinguish at least some target tutorial segments
- assign one label to each target tutorial segment under that factor
Prefer factors that are interpretable, and useful across many tutorial segments. Do not use factors based on narration, editing, presentation style, or simple restatements of segment content. If no valid factors can be identified reliably, return none and briefly explain why.

INPUTS
- Previous contextual factors:
{previous_facets}
- Signature of targets under previous contextual factors:
{signature}
- Targets:
{target_items}
- Tutorials:
{tutorials}
"""

# USER_PROMPT = """You are given tutorial videos and target information units extracted from them. Your job is to discover candidate contextual factors (facets) that help differentiate the procedural situations in which these information units occur.

# You are also given a previous factor set under which all targets share the same signature. Proposed factors must be distinct from that set.

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

# PROCEDURE

# 0. Exploratory search
# Generate 8-15 raw candidate factors spanning multiple families. For each, briefly note:
# - Title (2-3 words) and contextual family
# - One-sentence definition
# - Why it may separate some targets
# - Risks: overlap with previous or sibling candidates, ambiguity, low stability, too technical, target leakage

# 1. Candidate selection
# Select the best 3-6 factors that jointly maximize: validity (all five criteria), interpretability, target discrimination, family diversity, and distinctness from the previous set and from each other.

# 2. Final specification
# For each selected factor provide:
# - Context scope: global / mid-level / fine-grained
# - Title (2-3 words) and plain-language summary (one line a non-expert could understand)
# - Definition
# - Brief validity note: one sentence confirming it satisfies the five criteria
# - Why it separates at least some colliding targets
# - Why it is distinct from the previous set and from the other new factors
# - Segmentation guidelines: start/end rules, transcript or visual cues for label changes
# - Label vocabulary: finite, explicit, mutually exclusive, collectively exhaustive

# 3. Target assignments
# Assign one label per target per factor. For each, give a brief reason grounded in procedural context, not item content.

# 4. Anchor examples
# Per factor, provide 2-4 non-target items from the tutorial showing the factor applies broadly and multiple labels are used. Choose anchors with different labels from each other.

# 5. Signature check
# State which targets are now separated. If collisions remain, explain which and why.

# REFUSAL RULE
# Do NOT force artificial or item-specific factors. Return an empty candidate list with a refusal reason if, after searching across families and abstraction levels, no factor can differentiate the targets without encoding target content, relying on subjective judgment, or being too brittle to generalize.

# OUTPUT
# Return: selected factors, target assignments, anchor examples, signature verification, refusal reason (if applicable).

# INPUTS
# - Previous contextual factors:
# {previous_facets}
# - Signature of the target information units under the previous set of factors:
# {signature}
# - Target information units:
# {target_items}
# - Tutorials:
# {tutorials}
# """

# USER_PROMPT = """You are given tutorial videos and target information units extracted from them. Identify `contextual factors` (facets) that would help differentiate when these information units appear in the videos.

# A contextual factor is a domain-grounded, task-relevant, monolithic, piecewise-stable, and target-independent variable that describes the procedural situation of the task at a given moment or over a meaningful span.
#     - Domain-grounded: the factor is anchored in the task-world and describes a real aspect of the procedure as it unfolds, such as procedural stage, material state, tool state, object configuration, environmental condition, or service context. It does not describe the media, editorial, or presentation format through which the procedure is shown.
#     - Task-relevant: the factor changes what actions, constraints, interpretations, or information are applicable in that part of the task.
#     - Monolithic: the factor captures one coherent semantic concept rather than combining multiple distinct concepts.
#     - Piecewise-stable: the factor remains constant over meaningful temporal segments and changes at identifiable boundaries.
#     - Target-independent: the factor is determined by the broader procedural context, not by the specific information unit being labeled; removing that unit would not change the factor value.
# Non-examples of contextual factors include:
#     - video genre or tutorial format
#     - presenter configuration
#     - narration style
#     - camera or editing style
#     - entertainment framing
#     - human skill or expertise
#     - subjective quality judgments

# GOAL
# Propose contextual factors so that target information units receive DIFFERENT signatures (ideally unique). You are also given a previous set of factors under which all targets share the same signature, so avoid proposing factors that are similar or overlapping with the previous set.
# Follow the requirements and the procedure below.

# WHEN NOT TO PROPOSE FACTORS
# Do NOT force factors that are artificial or unreliable. Return an empty candidate list with a refusal reason if any of the following apply:
# - All target items occur in the same narrow region of the tutorial and share identical context — no factor can assign them different labels without reading the item content itself (violating target-independence).
# - The only distinguishing factors would be too fine-grained to generalize (e.g., "the exact product being discussed"), essentially re-encoding the item content as a label.
# - Proposed factors would require subjective or ambiguous judgment calls that different annotators would disagree on.
# It is better to return zero candidates than to propose a factor that does not reliably separate the targets.

# REQUIREMENTS
# - Each proposed factor must be distinct from the previous set of factors.
# - For each factor, labeling must be non-overlapping, exhaustive, and single-label (exactly one label per moment in time).
# - Keep factor titles short (2-3 words)
# - Provide a finite, explicit label vocabulary for each factor
# - Try to propose global or mid-level factors first. Available context scopes are:
#     - Global: stable across a large section or the whole tutorial.
#     - Mid-level: persists for a meaningful span of the tutorial.
#     - Fine-grained (only if necessary): brief contexts used only if global + mid-level factors cannot resolve the collisions.

# PROCEDURE
# 1. Propose contextual factors that improve discrimination among the target information units.
#     - Briefly justify: (1) why it is a contextual factor and satisfies each part of the definition, (2) how it separates (at least some of the) colliding information units, (3) why it is not contained in the previous set.
#     - Provide for each factor:
#         - Context scope ("global", "mid-level", "fine-grained")
#         - Title
#         - Definition
#         - Segmentation guidelines (start/end rules; transcript cues for label change)
#         - Full vocabulary necessary to segment the tutorial according to the factor.
#     - If no valid factors can be proposed, return an empty candidate list with a refusal reason.
# 2. If there are factors, assign a label to each TARGET information unit under each proposed factor.
#     - For each factor, provide a brief reason for each target assignment explaining why the label applies based on the surrounding context (not the item content itself).
#     - For each factor, also provide 2-4 anchor items (non-target items from the tutorial) that demonstrate the factor applies broadly and that different labels from the vocabulary are actually used across the tutorial. Choose anchors that receive DIFFERENT labels from each other to show the factor's range.
# 3. If there are factors, verify that the signatures of the target information units will be different under the new factors.

# OUTPUT
# Return the proposed contextual factors, target assignments, and anchor examples according to the schema.

# INPUTS
# - Previous contextual factors:
# {previous_facets}
# - Signature of the target information units under the previous set of factors:
# {signature}
# - Target information units:
# {target_items}
# - Tutorials:
# {tutorials}
# """


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
                USER_PROMPT,
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
            logger.error("Target segment %r not found in id_map; skipping.", item.id)
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
            temp_id = assignment.get("segment_id")  # LLM returns temp ID
            label = assignment.get("label")
            if temp_id not in rev_id_map:
                logger.error(
                    "Discovery hallucinated segment_id %r in target_assignments. Skipping.",
                    temp_id,
                )
                continue
            if label not in valid_labels:
                logger.error(
                    "Discovery hallucinated label %r for segment %r. Skipping.",
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
