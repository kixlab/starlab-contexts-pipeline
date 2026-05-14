"""
Formatting helpers for facet-related prompts.

Converts domain objects (Facet, FacetLabel, KnowledgeItem) into MessageContent
for inclusion in LLM prompts. Formatting logic lives here (not on domain classes)
so that display IDs can be controlled independently of real object IDs.
"""

from collections import Counter
from typing import Any, Dict, List, Optional

from utils import MessageContent

from facets.schema import Facet, FacetLabel, AnnotationTable
from knowledge_items.schema import KnowledgeBase, KnowledgeItem


# ============= ID mapping helpers =============

def build_item_id_map(items: List[KnowledgeItem]) -> Dict[str, str]:
    """Build mapping: real_id → sequential temp ID (I1, I2, ...).

    Both form and process functions receive the same items in the same order,
    so calling this in both gives a consistent, reversible mapping without
    passing extra state through batch_run_llm_calls.
    """
    return {item.id: f"S{i + 1}" for i, item in enumerate(items)}


def reverse_id_map(id_map: Dict[str, str]) -> Dict[str, str]:
    """Reverse a real→temp mapping to temp→real."""
    return {v: k for k, v in id_map.items()}


# ============= Text-level formatters (controllable display IDs) =============


_GUIDELINE_STR_FORMAT = """[{idx}] {guideline}"""

def format_guidelines(guidelines: List[str]) -> str:
    """Format the guidelines for the facet."""
    if len(guidelines) == 0:
        return "No guidelines are available."
    return "\n".join([_GUIDELINE_STR_FORMAT.format(idx=i+1, guideline=guideline) for i, guideline in enumerate(guidelines)])

_LABEL_STR_FORMAT = """<label><name>{label}</name><definition>{definition}</definition></label>"""
_LABEL_WITH_FREQUENCY_STR_FORMAT = """<label><name>{label}</name><definition>{definition}</definition><frequency>{frequency}</frequency></label>"""

def format_facet_label(label: FacetLabel, frequency: Optional[int] = None) -> str:
    if frequency is None:
        return _LABEL_STR_FORMAT.format(
            label=label.label,
            definition=label.definition,
        )
    return _LABEL_WITH_FREQUENCY_STR_FORMAT.format(
        label=label.label,
        definition=label.definition,
        frequency=frequency,
    )

def format_vocabulary(
    vocabulary: List[FacetLabel],
    label_frequencies: Optional[Dict[str, int]] = None,
) -> str:
    """Format the vocabulary for the facet."""
    if len(vocabulary) == 0:
        return "No labels are available."
    lines = [
        format_facet_label(
            label,
            label_frequencies.get(label.label, 0) if label_frequencies is not None else None,
        )
        for label in vocabulary
    ]
    if label_frequencies is not None:
        vocab_labels = {label.label for label in vocabulary}
        for label, frequency in sorted(label_frequencies.items()):
            if label not in vocab_labels:
                lines.append(
                    _LABEL_WITH_FREQUENCY_STR_FORMAT.format(
                        label=label,
                        definition="Observed annotation outside this vocabulary.",
                        frequency=frequency,
                    )
                )
    return "\n".join(lines)


def facet_label_frequencies(
    facet: Facet,
    knowledge_base: KnowledgeBase,
    annotations: AnnotationTable,
) -> Dict[str, int]:
    """Count raw item frequencies for a facet's labels in the active item scope."""
    counts = Counter({label.label: 0 for label in facet.vocabulary})
    for item in knowledge_base:
        if item.unit_id is None:
            continue
        label = annotations.by_item(item.id).get(facet.id, "na")
        counts[label] += 1
    return dict(counts)


_FACET_FORMAT = """<factor id="{display_id}">
<title>{title_plural}</title>
<definition>{definition}</definition>
<guidelines>
{guidelines}
</guidelines>
<vocabulary>
{vocabulary}
</vocabulary>
</factor>"""

def format_facet(
    facet: Facet,
    display_id: Optional[str] = None,
    label_frequencies: Optional[Dict[str, int]] = None,
) -> str:
    """Format a Facet as text using display_id instead of facet.id."""
    did = display_id if display_id is not None else facet.id
    return _FACET_FORMAT.format(
        display_id=did,
        title_plural=facet.title_plural,
        definition=facet.definition,
        guidelines=format_guidelines(facet.guidelines),
        vocabulary=format_vocabulary(facet.vocabulary, label_frequencies),
    )

_TUTORIAL_SEGMENT_FORMAT = """<segment id="{display_id}">
{text}
</segment>"""

def format_segment(
    segment: KnowledgeItem,
    display_id: Optional[str] = None,
    with_type: bool = True,
) -> str:
    """Format a KnowledgeItem as text, using display_id instead of real segment.id."""
    did = display_id if display_id is not None else segment.id
    text = segment.text
    if with_type and segment.subtype:
        subtype_val = segment.subtype.value if hasattr(segment.subtype, "value") else str(segment.subtype)
        text = f"({subtype_val}) {text}"
    return _TUTORIAL_SEGMENT_FORMAT.format(display_id=did, text=text)

# ============= MessageContent formatters =============


def facet_to_message_content(
    facet: Facet,
    display_id: Optional[str] = None,
    label_frequencies: Optional[Dict[str, int]] = None,
) -> MessageContent:
    """Format a single Facet as MessageContent."""
    return MessageContent(content=[{
        "type": "text",
        "text": format_facet(facet, display_id, label_frequencies),
    }])

def facet_display_id(index: int) -> str:
    """Get the display ID for a facet given its index in the list."""
    return f"F{index + 1}"

def facets_to_message_content(
    facets: List[Facet],
    annotations: Optional[AnnotationTable] = None,
    knowledge_base: Optional[KnowledgeBase] = None,
) -> MessageContent:
    """Format a list of Facets using sequential F1, F2 display IDs."""
    lines = []
    for i, facet in enumerate(facets):
        frequencies = None
        if annotations is not None and knowledge_base is not None:
            frequencies = facet_label_frequencies(facet, knowledge_base, annotations)
        lines.append(format_facet(facet, facet_display_id(i), frequencies))
    if len(lines) == 0:
        lines.append("No existing contextual factors.")
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
        lines.append(format_segment(item, display_id))
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
            lines.append(format_segment(item, display_id))
    return MessageContent(content=[{"type": "text", "text": "\n".join(lines)}])


_ANNOTATED_SEGMENT_FORMAT = """<segment id="{display_id}">
{text}
<factor_labels>{labels}</factor_labels>
</segment>"""

def format_segment_annotated(
    segment: KnowledgeItem,
    facets: List[Facet],
    annotations: AnnotationTable,
    display_id: Optional[str] = None,
    with_type: bool = True,
) -> str:
    """Format a KnowledgeItem with its pool annotation labels appended."""
    did = display_id if display_id is not None else segment.id
    text = segment.text
    if with_type and segment.subtype:
        subtype_val = segment.subtype.value if hasattr(segment.subtype, "value") else str(segment.subtype)
        text = f"({subtype_val}) {text}"
    labels_parts = [
        f"{facet_display_id(idx)}: {annotations[segment.id, facet.id]}"
        for idx, facet in enumerate(facets)
        if (segment.id, facet.id) in annotations
    ]
    labels_str = ", ".join(labels_parts) if labels_parts else ""
    return _ANNOTATED_SEGMENT_FORMAT.format(display_id=did, text=text, labels=labels_str)


def annotated_videos_to_message_content(
    videos: Dict[str, List[KnowledgeItem]],
    facets: List[Facet],
    annotations: AnnotationTable,
    id_map: Optional[Dict[str, str]] = None,
) -> MessageContent:
    """Like videos_to_message_content but appends per-facet pool labels to each segment."""
    lines = []
    for video_id, items in videos.items():
        lines.append(f"Video {video_id}:")
        for item in items:
            display_id = id_map[item.id] if id_map else None
            lines.append(format_segment_annotated(item, facets, annotations, display_id))
    if len(lines) == 0:
        lines.append("No tutorial segments.")
    return MessageContent(content=[{"type": "text", "text": "\n".join(lines)}])


def tree_stats_to_message_content(tree_stats: Dict[str, Any]) -> MessageContent:
    """Format refinement-tree statistics as MessageContent."""
    used = ", ".join(tree_stats.get("facets_used", [])) or "none"
    unused = ", ".join(tree_stats.get("facets_unused", [])) or "none"
    lines = [
        f"Total nodes: {tree_stats.get('total_nodes', 0)}",
        f"Contextual factors used in tree: {used}",
        f"Contextual factors not in tree: {unused}",
    ]
    return MessageContent(content=[{"type": "text", "text": "\n".join(lines)}])


def _format_combined_factors(
    schema: Any,
    facets: List[Facet],
    facet_display_ids: Dict[str, str],
    annotations: AnnotationTable,
    knowledge_base: KnowledgeBase,
    rejected_by_label: Optional[str] = None,
) -> str:
    lines = []
    lines.append("<factor>")
    schema_facet_ids = set(getattr(schema, "facet_ids", []))
    if not schema_facet_ids:
        lines.append("No factors.")
    for facet in facets:
        if facet.id not in schema_facet_ids:
            continue
        lines.append(format_facet(
            facet,
            display_id=facet_display_ids[facet.id],
            label_frequencies=facet_label_frequencies(facet, knowledge_base, annotations),
        ))
    
    if rejected_by_label is not None:
        lines.append(
            f'<rejected_by id="{rejected_by_label}"/>'
        )
        lines.append(
            "<rejection>"
            f"refinement_score={getattr(schema, 'refinement_score', 0.0):.4f}, "
            f"threshold={getattr(schema, 'refinement_threshold', 0.0):.4f}"
            "</rejection>"
        )
    lines.append("</factor>")
    return "\n".join(lines)

def schemas_to_message_content(
    accepted_schemas: List[Any],
    rejected_schemas: List[Any],
    facets: List[Facet],
    annotations: AnnotationTable,
    knowledge_base: KnowledgeBase,
) -> MessageContent:
    """Format accepted/rejected factor combinations with signature frequencies."""
    accepted_schemas = [
        schema for schema in accepted_schemas
        if getattr(schema, "facet_ids", [])
    ]
    accepted_label_by_id = {
        getattr(schema, "id", ""): f"A{idx}"
        for idx, schema in enumerate(accepted_schemas, start=1)
    }
    rejected_label_by_id = {
        getattr(schema, "id", ""): f"R{idx}"
        for idx, schema in enumerate(rejected_schemas, start=1)
    }
    lines = ["Accepted factors:"]
    if accepted_schemas:
        for schema in accepted_schemas:
            lines.append(_format_combined_factors(
                schema, facets, accepted_label_by_id, annotations, knowledge_base
            ))
    else:
        lines.append("No accepted factors.")

    lines.append("Rejected factors:")
    if rejected_schemas:
        for schema in rejected_schemas:
            rejected_by_id = getattr(schema, "rejected_by_id", "")
            rejected_by_label = accepted_label_by_id.get(rejected_by_id, "")
            lines.append(_format_combined_factors(
                schema, facets, rejected_label_by_id, annotations, knowledge_base, rejected_by_label=rejected_by_label
            ))
    else:
        lines.append("No rejected factors.")

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
