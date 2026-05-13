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
    FacetConsolidationResult,
)

from dataclasses import replace
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

PROPOSAL_USER_PROMPT = """You are given a contextual factor and a tutorial video. Propose vocabulary labels that capture the distinct values this factor takes in the tutorial.

A vocabulary label is a possible value of the given contextual factor. Labels should be grounded in the tutorial content, concise, and distinct from one another.

REQUIREMENTS
- Propose labels based only on values that are supported by the provided tutorial content.
- If the factor already has vocabulary labels, propose additional labels only when the provided content reveals values not covered by the existing vocabulary. Do not repeat or rename existing labels.
- Labels should be concise (1 to 3 words) and include a short definition.
- Labels should be mutually exclusive.
- If no new labels are needed, return an empty list.

INPUTS
- Contextual factor:
{facet}
- Tutorial content:
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

CONSOLIDATION_USER_PROMPT = """You are given a contextual factor with its current definition, guidelines, and vocabulary, plus additional vocabulary labels proposed from multiple tutorials. Your task is to consolidate the vocabulary. Then, adjust the definition and guidelines to stay consistent with the consolidated vocabulary.

CONSOLIDATE THE VOCABULARY
- Merge semantically equivalent or near-duplicate labels into a single canonical label.
- Prefer the most general stable vocabulary that preserves meaningful distinctions. Do not introduce new distinctions unless clearly supported by the evidence.
- Prefer existing facet vocabulary when adequate, and revise it only to reduce redundancy, ambiguity, or inconsistency.
- Keep the final labels mutually exclusive, interpretable, and at a single level of abstraction suitable for consistent segmentation.
- Use concise canonical names and track provenance by listing the original proposed labels that map to each final label in `source_labels`.
- When multiple valid vocabularies are possible, choose the one that is simplest, most reusable, and most consistent with the initial facet vocabulary.

REALIGN DEFINITION AND GUIDELINES
- Align the definition and guidelines with the consolidated vocabulary, while keeping the facet's scope and underlying meaning consistent. Make the minimal necessary revisions. 
OUTPUT
- `consolidated_labels`: for each label include canonical `label`, `definition`, and `source_labels`
- `revised_definition`: the (possibly unchanged) definition
- `revised_guidelines`: the (possibly unchanged) list of concise instructions
- `consolidation_notes`: brief summary of vocabulary consolidation decisions.

INPUTS
- Contextual factor with existing vocabulary:
{facet}
- Proposed labels by tutorials:
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
        "response_format": FacetConsolidationResult,
    }


def process_vocabulary_consolidation_response(
    response: Dict[str, Any],
    facet: Facet,
    **kwargs,
) -> Facet:
    """Process facet consolidation response, returning a revised Facet.

    The model may revise the definition and guidelines in addition to the vocabulary,
    so all three fields are replaced together. Identity-shaped fields (id, title,
    scope, justification, metadata) are preserved from the input facet.
    """
    result = response.get("response", {})
    labels = result.get("consolidated_labels", [])
    notes = result.get("consolidation_notes", "")
    logger.info("Consolidation notes for facet %s: %s", facet.id, notes)

    new_vocabulary = [
        FacetLabel(
            label=lbl.get("label", "").strip().lower(),
            definition=lbl.get("definition", "").strip(),
        )
        for lbl in labels
    ]

    revised_definition = (result.get("revised_definition") or facet.definition).strip()
    revised_guidelines_raw = result.get("revised_guidelines") or facet.guidelines
    revised_guidelines = [g.strip() for g in revised_guidelines_raw if g and g.strip()]

    return replace(
        facet,
        definition=revised_definition,
        guidelines=revised_guidelines,
        vocabulary=new_vocabulary,
    )