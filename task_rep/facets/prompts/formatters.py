"""
Formatting helpers for facet-related prompts.

Converts domain objects (Facet, FacetLabel, KnowledgeItem) into MessageContent
for inclusion in LLM prompts. Formatting logic lives here (not on domain classes)
so that display IDs can be controlled independently of real object IDs.
"""

from typing import Dict, List, Optional

from utils import MessageContent

from facets.schema import Facet, FacetLabel
from knowledge_items.schema import KnowledgeItem
from knowledge_items.prompts.formatters import format_item


# ============= ID mapping helpers =============

def build_item_id_map(items: List[KnowledgeItem]) -> Dict[str, str]:
    """Build mapping: real_id → sequential temp ID (I1, I2, ...).

    Both form and process functions receive the same items in the same order,
    so calling this in both gives a consistent, reversible mapping without
    passing extra state through batch_run_llm_calls.
    """
    return {item.id: f"I{i + 1}" for i, item in enumerate(items)}


def reverse_id_map(id_map: Dict[str, str]) -> Dict[str, str]:
    """Reverse a real→temp mapping to temp→real."""
    return {v: k for k, v in id_map.items()}


# ============= Text-level formatters (controllable display IDs) =============


_GUIDELINE_STR_FORMAT = """- {guideline}"""

def format_guidelines(guidelines: List[str]) -> str:
    """Format the guidelines for the facet."""
    if len(guidelines) == 0:
        return "No guidelines are available."
    return "\n".join([_GUIDELINE_STR_FORMAT.format(guideline=guideline) for guideline in guidelines])

_LABEL_STR_FORMAT = """[{label}] {definition}"""

def format_facet_label(label: FacetLabel) -> str:
    return _LABEL_STR_FORMAT.format(
        label=label.label,
        definition=label.definition,
    )

def format_vocabulary(vocabulary: List[FacetLabel]) -> str:
    """Format the vocabulary for the facet."""
    if len(vocabulary) == 0:
        return "No labels are available."
    return "\n".join([format_facet_label(label) for label in vocabulary])


_FACET_FORMAT = """[{display_id}] {title_plural}: {definition}
**Guidelines for labeling items**:
{guidelines}
**Vocabulary**:
{vocabulary}"""

def format_facet(facet: Facet, display_id: Optional[str] = None) -> str:
    """Format a Facet as text using display_id instead of facet.id."""
    did = display_id if display_id is not None else facet.id
    return _FACET_FORMAT.format(
        display_id=did,
        title_plural=facet.title_plural,
        definition=facet.definition,
        guidelines=format_guidelines(facet.guidelines),
        vocabulary=format_vocabulary(facet.vocabulary),
    )


# ============= MessageContent formatters =============


def facet_to_message_content(facet: Facet, display_id: Optional[str] = None) -> MessageContent:
    """Format a single Facet as MessageContent."""
    return MessageContent(content=[{"type": "text", "text": format_facet(facet, display_id)}])


def facets_to_message_content(facets: List[Facet]) -> MessageContent:
    """Format a list of Facets using sequential F1, F2 display IDs."""
    lines = []
    for i, facet in enumerate(facets):
        lines.append(format_facet(facet, f"F{i + 1}"))
    return MessageContent(content=[{"type": "text", "text": "\n".join(lines)}])


def knowledge_items_to_message_content(
    items: List[KnowledgeItem],
    id_map: Optional[Dict[str, str]] = None,
) -> MessageContent:
    """Format a list of KnowledgeItems as MessageContent.

    Args:
        id_map: Optional real_id → temp_id mapping. When provided, items are
                shown with short sequential IDs so the LLM can reproduce them.
    """
    lines = []
    for item in items:
        display_id = id_map[item.id] if id_map else None
        lines.append(format_item(item, display_id))
    return MessageContent(content=[{"type": "text", "text": "\n".join(lines)}])


def target_items_to_message_content(
    items: List[KnowledgeItem],
    id_map: Optional[Dict[str, str]] = None,
) -> MessageContent:
    """Format target KnowledgeItems as MessageContent.

    Args:
        id_map: Optional real_id → temp_id mapping for reliable LLM reproduction.
    """
    return knowledge_items_to_message_content(items, id_map)


def videos_to_message_content(
    videos: Dict[str, List[KnowledgeItem]],
    id_map: Optional[Dict[str, str]] = None,
) -> MessageContent:
    """Format a dict of video_id → items as MessageContent.

    Args:
        id_map: Optional real_id → temp_id mapping for reliable LLM reproduction.
    """
    lines = []
    for video_id, items in videos.items():
        lines.append(f"Video {video_id}:")
        for item in items:
            display_id = id_map[item.id] if id_map else None
            lines.append(format_item(item, display_id))
    return MessageContent(content=[{"type": "text", "text": "\n".join(lines)}])


def proposals_by_video_to_message_content(
    proposals_by_video: Dict[str, List[FacetLabel]],
) -> MessageContent:
    """Format per-video vocabulary proposals as MessageContent."""
    lines = []
    for video_id, proposals in proposals_by_video.items():
        lines.append(f"Video {video_id}:")
        for proposal in proposals:
            lines.append(format_facet_label(proposal))
    return MessageContent(content=[{"type": "text", "text": "\n".join(lines)}])
