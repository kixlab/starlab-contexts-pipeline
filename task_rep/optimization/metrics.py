"""
Core metrics for task schema evaluation.

Ports calc_discriminativeness, calc_compactness, and get_cell_to_units
from server/helpers/cim_scripts.py into the task_rep data model.
"""

from __future__ import annotations

import logging
import json
import math
from collections import defaultdict
from typing import Dict, List, Set, Union

from knowledge_items import (
    KnowledgeBase,
)
from facets import (
    Facet,
    AnnotationTable,
)

logger = logging.getLogger(__name__)

def compute_cells(
    knowledge_base: KnowledgeBase,
    facets: List[Facet],
    annotations: AnnotationTable,
) -> Dict[str, Dict[str, int]]:
    """
    Compute cell -> {unit_id -> item_count} mapping for a given schema.

    A cell is identified by the combined label signature across all
    facets for a given item. Items sharing the same signature belong
    to the same cell. Each cell maps unit_ids to the number of items
    from that unit in the cell.

    Items with no unit_id are skipped (not yet clustered).

    Port of server's get_cell_to_units() (cim_scripts.py:264-279).
    """
    cell_to_units: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for item in knowledge_base:
        if item.unit_id is None:
            continue
        subtype = item.subtype.value if hasattr(item.subtype, "value") else (str(item.subtype) if item.subtype is not None else None)
        signature = annotations.signature(item.id, facets, subtype=subtype)
        cell_to_units[signature][item.unit_id] += 1
    return {k: dict(v) for k, v in cell_to_units.items()}


def compute_discriminativeness(
    cell_to_units: Dict[str, Dict[str, int]],
    num_units: int = 0,
    base: int = 2,
) -> float:
    """
    Compute discriminativeness as H(unit|ctx) (lower is better).

    Uses item-level conditional entropy:
      H(U|C) = Σ_c P(C=c) · H(U|C=c)
    where:
      P(C=c)      = m_{•c} / m     (fraction of items in cell c)
      P(U=u|C=c)  = m_{u,c} / m_{•c}
      H(U|C=c)    = -Σ_u P(U=u|C=c) · log P(U=u|C=c)

    Interpretation:
      H = 0    → perfect separation (each cell has items from only one unit)
      H = H(U) → no separation (context provides no info about unit)

    Args:
        cell_to_units: Mapping of cell signature -> {unit_id -> item_count}.
        num_units: Unused, kept for backwards compatibility.
        base: Logarithm base.
    """
    m = sum(
        count
        for unit_counts in cell_to_units.values()
        for count in unit_counts.values()
    )
    if m == 0:
        return 0.0

    total = 0.0
    for unit_counts in cell_to_units.values():
        m_c = sum(unit_counts.values())
        if m_c == 0:
            continue
        p_c = m_c / m
        h_c = 0.0
        for count in unit_counts.values():
            if count > 0:
                p_u_given_c = count / m_c
                h_c -= p_u_given_c * math.log(p_u_given_c, base)
        total += p_c * h_c
    return total


def compute_compactness(facets: List[Facet]) -> int:
    """
    Compute compactness score C (lower is better).

    C = sum of vocabulary sizes across all facets, excluding 'na' labels.

    Port of server's calc_compactness() (cim_scripts.py:217-229),
    simplified to remove the initial_labels parameter (task_rep does not
    carry an information-type dimension as a separate baseline).
    """
    total = 0
    for facet in facets:
        for label in facet.vocabulary:
            if label.label.lower() != "na":
                total += 1
    return total


def count_unique_units(knowledge_base: KnowledgeBase) -> int:
    """Count distinct non-None unit_ids in the knowledge base."""
    return len({item.unit_id for item in knowledge_base if item.unit_id is not None})


def _population_variance(values: List[float]) -> float:
    """Compute population variance (biased, σ²) for a list of floats."""
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    return sum((x - mean) ** 2 for x in values) / len(values)


def compute_eta_squared(
    knowledge_base: KnowledgeBase,
    facets: List[Facet],
    annotations: AnnotationTable,
    position_mode: str = "ordinal",
    weighting: str = "equal",
) -> float:
    """
    Compute η² (eta-squared) averaged across videos.

    Measures how well a schema clusters items with the same signature
    at nearby positions within each tutorial video.

    η² ∈ [0, 1] where 1 = perfect within-tutorial clustering.

    Returns 1 − η² so the result has the same "lower is better"
    semantics as discriminativeness (0 = perfect, 1 = worst).

    Args:
        knowledge_base: Items with unit_ids assigned.
        facets: Subset of facets defining the schema.
        annotations: Label assignments.
        position_mode: "ordinal" uses source.local_index,
                       "time" uses midpoint of source.time_range.
        weighting: "equal" averages per-video η² equally,
                   "weighted" weights by item count per video.
    """
    # Group items by video
    by_video: Dict[str, List] = defaultdict(list)
    for item in knowledge_base:
        if item.unit_id is None:
            continue
        # Determine position
        if position_mode == "time":
            if item.source.time_range is None:
                continue
            pos = (item.source.time_range[0] + item.source.time_range[1]) / 2.0
        else:  # ordinal
            if item.source.local_index is None:
                continue
            pos = float(item.source.local_index)
        subtype = item.subtype.value if hasattr(item.subtype, "value") else (str(item.subtype) if item.subtype is not None else None)
        sig = annotations.signature(item.id, facets, subtype=subtype)
        by_video[item.source.video_id].append((pos, sig))

    eta_sum = 0.0
    weight_sum = 0.0

    for video_id, items in by_video.items():
        if len(items) <= 1:
            continue

        positions = [pos for pos, _ in items]
        pos_min, pos_max = min(positions), max(positions)
        if pos_min == pos_max:
            # All at same position → η² = 1.0 (trivially perfect)
            eta_video = 1.0
        else:
            # Normalize to [0, 1]
            norm = [(p - pos_min) / (pos_max - pos_min) for p in positions]
            total_var = _population_variance(norm)
            if total_var == 0:
                eta_video = 1.0
            else:
                # Group by signature
                groups: Dict[str, List[float]] = defaultdict(list)
                for (_, sig), n in zip(items, norm):
                    groups[sig].append(n)

                n_total = len(norm)
                within_var = sum(
                    (len(g) / n_total) * _population_variance(g)
                    for g in groups.values()
                )
                eta_video = 1.0 - within_var / total_var

        w = len(items) if weighting == "weighted" else 1.0
        eta_sum += eta_video * w
        weight_sum += w

    if weight_sum == 0:
        return 1.0  # No data → worst score

    eta_aggregate = eta_sum / weight_sum
    # Invert so lower = better (matches discriminativeness direction)
    return 1.0 - eta_aggregate


def compute_quality_metric(
    knowledge_base: KnowledgeBase,
    facets: List[Facet],
    annotations: AnnotationTable,
    metric_mode: str = "entropy",
    eta_squared_weighting: str = "equal",
) -> float:
    """
    Dispatch to the configured quality metric.

    Always returns a score where lower is better, regardless of mode.

    Args:
        metric_mode: "entropy" | "coherence_ordinal" | "coherence_temporal"
        eta_squared_weighting: "equal" | "weighted" (only for coherence modes)
    """
    if metric_mode == "entropy":
        cells = compute_cells(knowledge_base, facets, annotations)
        return compute_discriminativeness(cells)
    else:
        pos_mode = "ordinal" if metric_mode == "coherence_ordinal" else "time"
        return compute_eta_squared(
            knowledge_base, facets, annotations,
            position_mode=pos_mode, weighting=eta_squared_weighting,
        )
