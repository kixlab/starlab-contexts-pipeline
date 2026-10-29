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

def sample_target_items(
    knowledge_base: KnowledgeBase,
    cell_to_units: Dict[str, Set[str]],
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
        cell_to_units: Cell -> units mapping.
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