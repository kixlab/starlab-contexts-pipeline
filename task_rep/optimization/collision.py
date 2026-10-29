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

from ..knowledge_items.schema import KnowledgeBase, KnowledgeItem
from ..facets.schema import Facet, AnnotationTable


def find_colliding_cells(
    cell_to_units: Dict[str, Set[str]],
) -> Dict[str, Set[str]]:
    """
    Return cells that contain more than one unit (collisions).

    A collision means the current schema cannot distinguish between
    the units sharing the same cell signature.
    """
    return {sig: units for sig, units in cell_to_units.items() if len(units) > 1}


def sample_target_items(
    knowledge_base: KnowledgeBase,
    facets: List[Facet],
    annotations: AnnotationTable,
    colliding_cells: Dict[str, Set[str]],
    max_cells: int = 1,
    max_items_per_cell: int = 5,
) -> List[Tuple[List[str], str]]:
    """
    Sample representative items from colliding cells for facet discovery.

    For each selected cell, picks one item per unit to maximise diversity
    while avoiding intra-unit redundancy (items from the same unit are
    semantically equivalent by definition). These become the target_item_ids
    passed to VLMFacetExtractor.propose_facets().

    Port of target selection in server's update_facet_candidates_v2()
    (cim_scripts.py:381-420).

    Args:
        knowledge_base: All knowledge items.
        facets: Current schema facets.
        annotations: Current annotation table.
        colliding_cells: Output of find_colliding_cells().
        max_cells: Number of cells to sample from (largest first).
        max_items_per_cell: Maximum representative items per cell.

    Returns:
        List of (target_item_ids, cell_signature) tuples, one per cell.
        Each tuple represents one discovery request to the VLM.
    """
    if not colliding_cells:
        return []

    # Sort by collision size (largest first — most ambiguous cells first)
    sorted_cells = sorted(
        colliding_cells.items(),
        key=lambda kv: len(kv[1]),
        reverse=True,
    )

    # Build a lookup: cell_sig -> items in that cell
    # Must use the same subtype-prefixed signature as compute_cells so keys match.
    cell_items: Dict[str, List[KnowledgeItem]] = defaultdict(list)
    for item in knowledge_base:
        if item.unit_id is None:
            continue
        subtype = item.subtype.value if hasattr(item.subtype, "value") else (str(item.subtype) if item.subtype is not None else None)
        sig = annotations.signature(item.id, facets, subtype=subtype)
        cell_items[sig].append(item)

    results = []
    for cell_sig, _unit_ids in sorted_cells[:max_cells]:
        items_in_cell = cell_items.get(cell_sig, [])

        # Group by unit — pick one representative per unit
        by_unit: Dict[str, List[KnowledgeItem]] = defaultdict(list)
        for item in items_in_cell:
            by_unit[item.unit_id].append(item)

        representatives = [random.choice(unit_items) for unit_items in by_unit.values()]

        if len(representatives) > max_items_per_cell:
            representatives = random.sample(representatives, max_items_per_cell)

        # Need at least 2 items to have something to discriminate
        if len(representatives) < 2:
            continue

        results.append(([item.id for item in representatives], cell_sig))

    return results
