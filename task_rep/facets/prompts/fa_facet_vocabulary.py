"""
Prompts for building facet vocabularies from knowledge items.

Two-phase process:
1. Per-(facet, video) proposal: Propose vocabulary labels for one facet
   based on knowledge items from one video. Batched across all (facet, video) pairs.
2. Per-facet consolidation: Merge and deduplicate proposals across videos.
   Batched across all facets.
"""

import logging
from typing import Any, Dict, List

from utils.llm import MessageContent

from .fa_pydantic import (
    VocabularyProposalResult,
    VocabularyConsolidationResult,
)
from .formatters import (
    facet_to_message_content,
    knowledge_items_to_message_content,
    proposals_by_video_to_message_content,
    build_item_id_map,
)

from facets.schema import Facet, FacetLabel
from knowledge_items.schema import KnowledgeItem

logger = logging.getLogger(__name__)

# ============= Phase 1: Per-(Facet, Video) Proposal =============

PROPOSAL_SYSTEM_PROMPT = """You are an expert at analyzing tutorial content and identifying vocabulary labels for contextual factors related to the task: {task}. Inputs are structured with XML tags. Use tag names and id attributes to identify objects and boundaries."""

PROPOSAL_USER_PROMPT = """You are given a contextual factor and tutorial segments extracted from a tutorial video. Propose vocabulary labels that capture the distinct values this factor takes in the provided content.

A vocabulary label is a possible value of the given contextual factor. Labels should be grounded in the procedural content, concise, and distinct from one another.

REQUIREMENTS
- Propose labels based only on values that are supported by the provided tutorial segments.
- Labels should be concise (2 to 5 words) and include a short definition.
- Labels should be mutually exclusive where possible.
- If the factor already has vocabulary labels, propose additional labels only when the provided content reveals values not covered by the existing vocabulary.
- Do not repeat or rename existing labels.
- If no new labels are needed, return an empty list.

INPUTS
- Contextual factor:
{facet}
- Tutorial segments from this video:
{video_items}"""


def form_vocabulary_proposal_request(
    task: str,
    facet: Facet,
    video_items: List[KnowledgeItem],
    **kwargs,
) -> Dict[str, Any]:
    """Form request to propose vocabulary labels for one facet from one video."""
    id_map = build_item_id_map(video_items)
    messages = [
        {
            "role": "system",
            "content": MessageContent.from_template(PROPOSAL_SYSTEM_PROMPT, task=task),
        },
        {
            "role": "user",
            "content": MessageContent.from_template(
                PROPOSAL_USER_PROMPT,
                facet=facet_to_message_content(facet, "F"),
                video_items=knowledge_items_to_message_content(video_items, id_map),
            ),
        },
    ]
    return {
        "messages": messages,
        "response_format": VocabularyProposalResult,
    }


def process_vocabulary_proposal_response(
    response: Dict[str, Any],
    **kwargs,
) -> List[FacetLabel]:
    """Process per-(facet, video) vocabulary proposal, returning typed FacetLabels."""
    result = response.get("response", {})
    proposed_labels = result.get("proposed_labels", [])
    return [
        FacetLabel(
            label=lbl.get("label", "").strip().lower(),
            definition=lbl.get("definition", "").strip(),
        )
        for lbl in proposed_labels
    ]


# ============= Phase 2: Per-Facet Consolidation =============

CONSOLIDATION_SYSTEM_PROMPT = """You are an expert at organizing and consolidating vocabulary labels for contextual factors related to the task: {task}. Inputs are structured with XML tags. Use tag names and id attributes to identify objects and boundaries."""

CONSOLIDATION_USER_PROMPT = """You are given a contextual factor with its definition and guidelines, along with vocabulary labels proposed from multiple tutorial videos. Your task is to consolidate these into a single, clean vocabulary for segmentation.

REQUIREMENTS
- Merge semantically equivalent or near-duplicate labels into a single canonical label.
- Prefer the most general stable vocabulary that preserves the distinctions needed for segmentation. Do not introduce new distinctions unless clearly supported by the evidence.
- Prefer existing facet vocabulary when adequate, and revise it only to reduce redundancy, ambiguity, or inconsistency.
- Keep the final labels mutually exclusive, interpretable, and at a single level of abstraction suitable for consistent segmentation.
- Use concise canonical names and track provenance by listing the original proposed labels that map to each final label in `source_labels`.
- When multiple valid vocabularies are possible, choose the one that is simplest, most reusable, and most consistent with the existing facet vocabulary.

OUTPUT
Return the final consolidated vocabulary. For each label, include:
- canonical label
- definition
- source_labels

INPUTS
- Facet:
{facet}
- Proposed labels by video:
{proposals_by_video}
"""


def form_vocabulary_consolidation_request(
    task: str,
    facet: Facet,
    proposals_by_video: Dict[str, List[FacetLabel]],
    **kwargs,
) -> Dict[str, Any]:
    """Form request to consolidate vocabulary labels for a single facet."""
    messages = [
        {
            "role": "system",
            "content": MessageContent.from_template(CONSOLIDATION_SYSTEM_PROMPT, task=task),
        },
        {
            "role": "user",
            "content": MessageContent.from_template(
                CONSOLIDATION_USER_PROMPT,
                facet=facet_to_message_content(facet, "F"),
                proposals_by_video=proposals_by_video_to_message_content(proposals_by_video),
            ),
        },
    ]
    return {
        "messages": messages,
        "response_format": VocabularyConsolidationResult,
    }


def process_vocabulary_consolidation_response(
    response: Dict[str, Any],
    **kwargs,
) -> List[FacetLabel]:
    """Process vocabulary consolidation response, returning typed FacetLabels."""
    result = response.get("response", {})
    labels = result.get("consolidated_labels", [])
    notes = result.get("consolidation_notes", "")
    logger.info("Consolidation notes: %s", notes)
    return [
        FacetLabel(
            label=lbl.get("label", "").strip().lower(),
            definition=lbl.get("definition", "").strip(),
        )
        for lbl in labels
    ]