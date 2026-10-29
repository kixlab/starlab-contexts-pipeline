"""
Prompts for building facet vocabularies from knowledge items.

Two-phase process:
1. Per-(facet, video) proposal: Propose vocabulary labels for one facet
   based on knowledge items from one video. Batched across all (facet, video) pairs.
2. Per-facet consolidation: Merge and deduplicate proposals across videos.
   Batched across all facets.
"""

from typing import Any, Dict, List

from ..utils.llm import MessageContent

from .fa_pydantic import (
    VocabularyProposalResult,
    VocabularyConsolidationResult,
)

from .formatters import list_mc_to_str


# ============= Phase 1: Per-(Facet, Video) Proposal =============

PROPOSAL_SYSTEM_PROMPT = """You are an expert at analyzing tutorial content and identifying vocabulary labels for contextual factors related to the task: {task}."""

PROPOSAL_USER_PROMPT = """You are given a contextual factor (facet) and knowledge items extracted from a tutorial video. Propose vocabulary labels that capture the distinct values this facet takes within the video content.

DEFINITIONS
- A contextual factor (facet) is a piecewise-stable, target-independent, task-relevant dimension that discriminates knowledge items.
- A vocabulary label is one possible value a facet can take. Labels must be mutually exclusive and collectively cover all content relevant to the facet.

REQUIREMENTS
- Propose labels that are grounded in the actual video content provided.
- Labels should be concise (2-5 words) with clear definitions.
- If the facet already has vocabulary labels, propose ADDITIONAL labels only if the video content reveals values not covered by the existing vocabulary. Do not re-propose existing labels.
- If no new labels are needed, return an empty list.

INPUTS
- Facet:
{facet}
- Knowledge items from this video:
{video_items}
"""


def form_vocabulary_proposal_request(
    task: str,
    facet: MessageContent,
    video_items: List[MessageContent],
    **kwargs,
) -> Dict[str, Any]:
    """Form request to propose vocabulary labels for one facet from one video."""
    video_items_str = list_mc_to_str(video_items)

    messages = [
        {
            "role": "system",
            "content": MessageContent.from_template(
                PROPOSAL_SYSTEM_PROMPT,
                task=task,
            )
        },
        {
            "role": "user",
            "content": MessageContent.from_template(
                PROPOSAL_USER_PROMPT,
                facet=facet,
                video_items=video_items_str,
            )
        }
    ]
    return {
        "messages": messages,
        "response_format": VocabularyProposalResult,
    }


def process_vocabulary_proposal_response(
    response: Dict[str, Any],
    **kwargs,
) -> List[Dict[str, Any]]:
    """Process response from per-(facet, video) vocabulary proposal.

    Returns:
        List of dicts: [{"label": str, "definition": str}]
    """
    result = response.get("response", {})
    proposed_labels = result.get("proposed_labels", [])

    return [
        {
            "label": lbl.get("label", "").strip(),
            "definition": lbl.get("definition", "").strip(),
        }
        for lbl in proposed_labels
    ]


# ============= Phase 2: Per-Facet Consolidation =============

CONSOLIDATION_SYSTEM_PROMPT = """You are an expert at organizing and consolidating vocabulary labels for contextual factors related to the task: {task}."""

CONSOLIDATION_USER_PROMPT = """You are given a contextual factor (facet) with its definition and guidelines, along with vocabulary labels proposed from multiple tutorial videos. Your goal is to consolidate all labels into a single, clean vocabulary.

REQUIREMENTS
1. Merge semantically equivalent labels: If two or more proposed labels describe the same concept (even with different wording), merge them into a single canonical label.
2. Preserve existing vocabulary: Labels already in the facet vocabulary should be kept unless they are clearly redundant with other existing labels.
3. Maintain mutual exclusivity: The final vocabulary should consist of non-overlapping labels.
4. Ensure completeness: The final vocabulary should cover all distinct values observed across the corpus.
5. Use clear canonical names: Each label should have a concise name (2-5 words) and a definition broad enough to cover variations but specific enough to distinguish from other labels.
6. Track provenance: For each consolidated label, list the original proposed labels that were merged into it (in source_labels).

INPUTS
- Facet:
{facet}
- Existing vocabulary:
{existing_vocabulary}
- Proposed labels by video:
{proposals_by_video}
"""


def form_vocabulary_consolidation_request(
    task: str,
    facet: MessageContent,
    existing_vocabulary: str,
    proposals_by_video: MessageContent,
    **kwargs,
) -> Dict[str, Any]:
    """Form request to consolidate vocabulary labels for a single facet."""
    messages = [
        {
            "role": "system",
            "content": MessageContent.from_template(
                CONSOLIDATION_SYSTEM_PROMPT,
                task=task,
            )
        },
        {
            "role": "user",
            "content": MessageContent.from_template(
                CONSOLIDATION_USER_PROMPT,
                facet=facet,
                existing_vocabulary=existing_vocabulary,
                proposals_by_video=proposals_by_video,
            )
        }
    ]
    return {
        "messages": messages,
        "response_format": VocabularyConsolidationResult,
    }


def process_vocabulary_consolidation_response(
    response: Dict[str, Any],
    **kwargs,
) -> List[Dict[str, Any]]:
    """Process response from vocabulary consolidation.

    Returns:
        List of dicts: [{"label": str, "definition": str, "source_labels": [str]}]
    """
    result = response.get("response", {})
    labels = result.get("consolidated_labels", [])

    return [
        {
            "label": lbl.get("label", "").strip(),
            "definition": lbl.get("definition", "").strip(),
            "source_labels": lbl.get("source_labels", []),
        }
        for lbl in labels
    ]
