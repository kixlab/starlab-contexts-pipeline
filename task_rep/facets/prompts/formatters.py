"""
Formatting helpers for facet-related prompts.

Converts domain objects (Facet, FacetLabel, KnowledgeItem) into MessageContent
for inclusion in LLM prompts. Formatting logic lives here (not on domain classes)
so that display IDs can be controlled independently of real object IDs.
"""

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

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
<title>{title_plural}</title>{kind}
<user_question>{user_question}</user_question>
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
        kind=f"\n<kind>{facet.kind}</kind>" if facet.kind else "",
        user_question=facet.user_question or "No user-facing question is available.",
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


_HISTORY_ENTRY_FORMAT = """<tried_factor><title>{title}</title><scope>{scope}</scope><definition>{definition}</definition><outcome>{outcome}</outcome></tried_factor>"""

_DESIGN_SPACE_GUIDE = """HOW THE DESIGN SPACE IS BUILT
- A combination is {combos}. It splits the segments into cells: one cell for each combination of values that occurs.
- Overlap: how much of a combination's information another combination already carries (their shared information divided by the combination's own information). 100% means the other combination fully determines it; 0% means they are unrelated.
- Combinations are checked from fewest values to most. A combination is kept unless an already-kept combination with no more values has an overlap above {limit} with it; otherwise it is pruned.
- A kept combination B refines a kept combination A when B carries more than {limit} of A's information and has more values.
- The design space is rebuilt from scratch after every submission, so earlier decisions can change."""


def schemas_to_message_content(
    accepted_schemas: List[Any],
    rejected_schemas: List[Any],
    facets: List[Facet],
    annotations: AnnotationTable,
    knowledge_base: KnowledgeBase,
    refinement_edges: Optional[List[Tuple[str, str]]] = None,
    refinement_threshold: Optional[float] = None,
    last_submission: Optional[Any] = None,
) -> MessageContent:
    """Format the design space (DAG) of factor combinations.

    Shows how it is built, each factor once, the kept combinations (why each
    was kept and what it refines), the pruned single factors (and whether they
    are used anywhere), one summary line for pruned pairs, and what the last
    submission changed. Linear in the number of factors and kept
    combinations; nothing grows with the number of segments.

    ``last_submission`` needs ``title``, ``facet_id``, ``outcome``,
    ``entered`` and ``previous_schema_ids`` (see ``contextual.feedback``).
    """
    display = {facet.id: facet_display_id(idx) for idx, facet in enumerate(facets)}
    order = {facet.id: idx for idx, facet in enumerate(facets)}
    by_id = {facet.id: facet for facet in facets}

    def rank(facet_ids) -> List[int]:
        return sorted(order[fid] for fid in facet_ids if fid in order)

    def factor(fid: str) -> str:
        return f"{display[fid]} ({by_id[fid].title_plural or by_id[fid].title})"

    def name(facet_ids) -> str:
        ids = sorted((fid for fid in facet_ids if fid in by_id), key=order.__getitem__)
        return " + ".join(factor(fid) for fid in ids) if ids else "(no factors)"

    def cell_count(facet_ids) -> int:
        node_facets = [by_id[fid] for fid in facet_ids if fid in by_id]
        return len({
            annotations.signature(item.id, node_facets)
            for item in knowledge_base if item.unit_id is not None
        })

    tau = refinement_threshold
    if tau is None and rejected_schemas:
        tau = getattr(rejected_schemas[0], "refinement_threshold", None)
    limit = f"{tau:.0%}" if tau is not None else "the limit"
    all_schemas = list(accepted_schemas) + list(rejected_schemas)
    max_facets = max((len(s.facet_ids) for s in all_schemas), default=1)
    with_pairs = max_facets >= 2

    kept = sorted((s for s in accepted_schemas if s.facet_ids),
                  key=lambda s: (len(s.facet_ids), rank(s.facet_ids)))
    labels = {s.id: f"C{i}" for i, s in enumerate(kept, start=1)}

    def ref(schema_id: str, facet_ids) -> str:
        return f"{labels.get(schema_id, 'a kept combination')} ({name(facet_ids)})"

    parents: Dict[str, List[str]] = {}
    for source, target in refinement_edges or []:
        if source in labels and target in labels:
            parents.setdefault(target, []).append(labels[source])

    lines = [_DESIGN_SPACE_GUIDE.format(
        combos="a single factor or a pair of factors" if with_pairs else "a single factor",
        limit=limit), "", "Factors:"]
    lines += [b["text"] for b in facets_to_message_content(
        facets, annotations=annotations, knowledge_base=knowledge_base).content]

    kept_singles = [s for s in kept if len(s.facet_ids) == 1]
    kept_pairs = [s for s in kept if len(s.facet_ids) == 2]
    all_pairs = [s for s in all_schemas if len(s.facet_ids) == 2]
    summary = f"Design space: kept {len(kept_singles)} of {len(facets)} single factors"
    summary += (f" and {len(kept_pairs)} of {len(all_pairs)} pairs." if with_pairs else ".")
    lines += ["", summary, "", "Kept combinations:"]
    for schema in kept:
        if len(schema.facet_ids) > 1 and getattr(schema, "member_overlaps", None):
            ids = sorted(schema.member_overlaps, key=lambda f: order.get(f, 0))
            why = ", ".join(f"{display.get(f, f)} alone carries "
                            f"{schema.member_overlaps[f]:.0%}" for f in ids)
            closest = getattr(schema, "closest_id", "")
            if closest and set(getattr(schema, "closest_facet_ids", [])) - set(schema.facet_ids):
                why += (f"; the most any other kept combination carries is "
                        f"{schema.closest_overlap:.0%} ({labels.get(closest, '?')})")
        elif getattr(schema, "closest_id", ""):
            why = (f"the most any kept combination with no more values carries is "
                   f"{schema.closest_overlap:.0%} ({ref(schema.closest_id, schema.closest_facet_ids)})")
        else:
            why = "no kept combination has fewer values"
        refines = parents.get(schema.id)
        lines.append(
            f"- {labels[schema.id]}: {name(schema.facet_ids)}, "
            f"{cell_count(schema.facet_ids)} cells; kept because {why}"
            + (f"; refines {', '.join(sorted(refines))}" if refines else "")
        )
    if not kept:
        lines.append("- none yet")

    lines += ["", "Pruned single factors:"]
    pruned_singles = sorted((s for s in rejected_schemas if len(s.facet_ids) == 1),
                            key=lambda s: rank(s.facet_ids))
    for schema in pruned_singles:
        fid = schema.facet_ids[0]
        used_in = [labels[k.id] for k in kept if fid in k.facet_ids]
        lines.append(
            f"- {name(schema.facet_ids)}: pruned; "
            f"{ref(schema.rejected_by_id, schema.rejected_by_facet_ids)} already carries "
            f"{schema.refinement_score:.0%} of its information; "
            + (f"still used in {', '.join(used_in)}" if used_in else "used nowhere")
        )
    if not pruned_singles:
        lines.append("- none")

    if with_pairs:
        pruned_pairs = [s for s in rejected_schemas if len(s.facet_ids) == 2]
        own = sum(1 for s in pruned_pairs
                  if set(s.rejected_by_facet_ids) <= set(s.facet_ids))
        line = f"Pruned pairs: {len(pruned_pairs)} of {len(all_pairs)}."
        if pruned_pairs:
            line += (f" In {own}, one of the pair's own factors already carries more than "
                     f"{limit} of it; in {len(pruned_pairs) - own}, another kept "
                     f"combination does.")
        lines += ["", line]

    if last_submission is not None:
        lines += ["", "Last submission:"]
        lines += _last_submission_lines(
            last_submission, kept, rejected_schemas, labels, display, name, ref, with_pairs)
    return MessageContent(content=[{"type": "text", "text": "\n".join(lines)}])


def _last_submission_lines(sub, kept, rejected, labels, display, name, ref, with_pairs) -> List[str]:
    title = getattr(sub, "title", "")
    fid = getattr(sub, "facet_id", "")
    head = (f"- \"{title}\"" + (f" ({display[fid]})" if fid in display else "") + ": "
            if title else "- ")
    head += f"{getattr(sub, 'outcome', '')}."
    if not getattr(sub, "entered", False):
        return [head + (" It did not enter the design space." if title else "")]
    lines = [head]
    single_kept = next((s for s in kept if s.facet_ids == [fid]), None)
    single_pruned = next((s for s in rejected if s.facet_ids == [fid]), None)
    if single_kept is not None:
        lines.append(f"- As a single factor: kept as {labels[single_kept.id]}.")
    elif single_pruned is not None:
        lines.append(
            f"- As a single factor: pruned; "
            f"{ref(single_pruned.rejected_by_id, single_pruned.rejected_by_facet_ids)} "
            f"already carries {single_pruned.refinement_score:.0%} of its information.")
    if with_pairs:
        kept_pairs = [s for s in kept if len(s.facet_ids) == 2 and fid in s.facet_ids]
        pruned_pairs = [s for s in rejected if len(s.facet_ids) == 2 and fid in s.facet_ids]
        if kept_pairs:
            partners = ", ".join(
                f"{labels[s.id]} (with {name([f for f in s.facet_ids if f != fid])})"
                for s in kept_pairs)
            lines.append(f"- In pairs: kept in {partners}; {len(pruned_pairs)} other pairs pruned.")
        else:
            lines.append(f"- In pairs: none of its {len(pruned_pairs)} pairs is kept.")
    before = set(getattr(sub, "previous_schema_ids", []))
    changes = [
        f"{name(s.facet_ids)} is now pruned; "
        f"{ref(s.rejected_by_id, s.rejected_by_facet_ids)} carries {s.refinement_score:.0%} of it"
        for s in rejected if s.id in before
    ] + [
        f"{labels[s.id]} ({name(s.facet_ids)}) is now kept"
        for s in kept if s.id not in before and fid not in s.facet_ids
    ]
    lines.append("- Elsewhere: " + ("; ".join(changes) + "." if changes
                                    else "nothing else in the design space changed."))
    return lines


def history_to_message_content(history: List[Any]) -> MessageContent:
    """Format previously tried, non-kept factors with their outcomes.

    Each entry needs ``title``, ``scope``, ``definition``, ``outcome`` and an
    optional ``duplicate_of`` (title of the matched factor).
    """
    lines = []
    for entry in history:
        outcome = getattr(entry, "outcome", "")
        duplicate_of = getattr(entry, "duplicate_of", None)
        if duplicate_of:
            outcome = f"{outcome} of {duplicate_of}"
        lines.append(_HISTORY_ENTRY_FORMAT.format(
            title=getattr(entry, "title", ""),
            scope=getattr(entry, "scope", ""),
            definition=getattr(entry, "definition", ""),
            outcome=outcome,
        ))
    if not lines:
        lines.append("None yet.")
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
