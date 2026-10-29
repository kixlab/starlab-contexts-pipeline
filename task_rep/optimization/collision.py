"""
Collision detection and target item sampling for facet discovery.

A collision is a cell containing more than one unit — the current schema
cannot tell those units apart. The algorithm resolves collisions by
discovering new facets that separate them.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Set, Tuple

from knowledge_items import (
    KnowledgeBase,
    KnowledgeItem,
)
from facets import (
    Facet,
    AnnotationTable,
)
from .metrics import compute_cells


def sample_target_items(
    knowledge_base: KnowledgeBase,
    facets: List[Facet],
    annotations: AnnotationTable,
    max_cells: int = 1,
    max_items_per_cell: int = 5,
) -> List[Tuple[List[str], str]]:
    """
    Sample representative items from cells for facet discovery.

    For each selected cell, picks one item per unit to maximise diversity
    while avoiding intra-unit redundancy (items from the same unit are
    semantically equivalent by definition). These become the target_item_ids
    passed to VLMFacetExtractor.propose_facets().

    Port of target selection in server's update_facet_candidates_v2()
    (cim_scripts.py:381-420).

    Args:
        knowledge_base: All knowledge items.
        facets: Current schema facets.
        annotations: Current label assignments.
        max_cells: Number of cells to sample from (largest first).
        max_items_per_cell: Maximum representative items per cell.

    Returns:
        List of (target_item_ids, cell_signature) tuples, one per cell with at least 2 units.
        Each tuple represents one discovery request to the VLM.
    """
    if max_cells < 1:
        return []
    if max_items_per_cell < 2:
        return []

    cell_to_units = compute_cells(knowledge_base, facets, annotations)

    # Sort by cell size (largest first — most ambiguous cells first)
    sorted_cells = sorted(
        cell_to_units.items(),
        key=lambda kv: len(kv[1]),
        reverse=True,
    )

    # Build a lookup: unit_id -> items in that unit
    unit_id_to_items: Dict[str, List[KnowledgeItem]] = defaultdict(list)
    for item in knowledge_base:
        if item.unit_id is None:
            continue
        unit_id_to_items[item.unit_id].append(item)

    results = []
    for cell_sig, _unit_ids in sorted_cells[:max_cells]:
        if len(_unit_ids) < 2:
            continue

        representatives = []
        for _unit_id in _unit_ids:
            if len(unit_id_to_items[_unit_id]) == 0:
                continue
            rep = random.choice(unit_id_to_items[_unit_id])
            representatives.append(rep)
        
        if len(representatives) < 2:
            continue

        if len(representatives) > max_items_per_cell:
            representatives = random.sample(representatives, max_items_per_cell)

        results.append(([item.id for item in representatives], cell_sig))

    return results


def sample_target_items_coherence(
    knowledge_base: KnowledgeBase,
    facets: List[Facet],
    annotations: AnnotationTable,
    position_mode: str = "ordinal",
    max_cells: int = 1,
) -> List[Tuple[List[str], str]]:
    """
    Sample target items for facet discovery in coherence (eta-squared) mode.

    Finds within-tutorial signature groups with high positional spread
    and selects collision sites using weighted random sampling.  Groups
    with larger spread are more likely to be picked, but randomness
    avoids overfitting to a single tutorial's idiosyncrasies.

    All items in a selected group are returned (they share the same
    tutorial, so there is no need to subsample).

    Args:
        knowledge_base: All knowledge items.
        facets: Current schema facets.
        annotations: Current label assignments.
        position_mode: "ordinal" (local_index) or "time" (time_range midpoint).
        max_cells: Number of collision sites to sample.

    Returns:
        List of (target_item_ids, signature) tuples.
    """
    if max_cells < 1:
        return []

    # Build per-(video, signature) groups with positions
    GroupEntry = Tuple[str, float]  # (item_id, position)
    groups: Dict[Tuple[str, str], List[GroupEntry]] = defaultdict(list)

    for item in knowledge_base:
        if item.unit_id is None:
            continue
        # Determine position
        if position_mode == "time":
            if item.source.time_range is None:
                continue
            pos = (item.source.time_range[0] + item.source.time_range[1]) / 2.0
        else:
            if item.source.local_index is None:
                continue
            pos = float(item.source.local_index)

        subtype = item.subtype.value if hasattr(item.subtype, "value") else (str(item.subtype) if item.subtype is not None else None)
        sig = annotations.signature(item.id, facets, subtype=subtype)
        groups[(item.source.video_id, sig)].append((item.id, pos))

    # Score each group by positional spread (variance)
    candidates: List[Tuple[Tuple[str, str], List[GroupEntry], float]] = []
    for key, entries in groups.items():
        if len(entries) < 2:
            continue
        positions = [pos for _, pos in entries]
        pos_min, pos_max = min(positions), max(positions)
        if pos_min == pos_max:
            continue  # no spread
        # Normalize then compute variance
        norm = [(p - pos_min) / (pos_max - pos_min) for p in positions]
        mean = sum(norm) / len(norm)
        variance = sum((x - mean) ** 2 for x in norm) / len(norm)
        if variance > 0:
            candidates.append((key, entries, variance))

    if not candidates:
        return []

    # Weighted random selection: bigger spread → more likely
    weights = [var for _, _, var in candidates]
    k = min(max_cells, len(candidates))
    selected = random.choices(candidates, weights=weights, k=k)

    results = []
    for (video_id, sig), entries, _ in selected:
        item_ids = [item_id for item_id, _ in entries]
        results.append((item_ids, sig))

    return results