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
from typing import Dict, List, Set

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
) -> Dict[str, Set[str]]:
    """
    Compute cell -> unit_ids mapping for a given schema.

    A cell is identified by the combined label signature across all
    facets for a given item. Items sharing the same signature belong
    to the same cell. The cell maps to the set of unit_ids it contains
    (not item_ids, since multiple items can belong to the same unit).

    Items with no unit_id are skipped (not yet clustered).

    Port of server's get_cell_to_units() (cim_scripts.py:264-279).
    """
    cell_to_units: Dict[str, Set[str]] = defaultdict(set)
    for item in knowledge_base:
        if item.unit_id is None:
            continue
        subtype = item.subtype.value if hasattr(item.subtype, "value") else (str(item.subtype) if item.subtype is not None else None)
        signature = annotations.signature(item.id, facets, subtype=subtype)
        cell_to_units[signature].add(item.unit_id)
    return dict(cell_to_units)


def compute_discriminativeness(
    cell_to_units: Dict[str, Set[str]],
    num_units: int,
    base: int = 2,
) -> float:
    """
    Compute discriminativeness score D (lower is better).

    D = sum_c (n_c / N) * log_base(n_c)
    where n_c = units in cell c, N = total distinct units.

    Interpretation:
      D = 0           → perfect separation (each unit in its own cell)
      D = log_base(N) → no separation (all units in one cell)

    """
    if num_units == 0:
        return 0.0
    total = 0.0
    for unit_set in cell_to_units.values():
        n_c = len(unit_set)
        if n_c > 0:
            total += (n_c / num_units) * math.log(n_c, base)
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
