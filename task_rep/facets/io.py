"""
I/O utilities for facets and annotations.

Handles saving, loading, and cache-aware skipping for:
- Facets (with vocabulary)
- AnnotationTable
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Set

from .schema import AnnotationTable, Facet


# ---- save / load: facets ----

def save_facets(facets: List[Facet], path: str | Path) -> None:
    """Save facets to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([facet.to_dict() for facet in facets], f, ensure_ascii=False, indent=2)


def load_facets(path: str | Path) -> List[Facet]:
    """Load facets from a JSON file. Returns empty list if file doesn't exist."""
    path = Path(path)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [Facet.from_dict(d) for d in json.load(f)]


# ---- save / load: annotations ----

def save_annotations(annotations: AnnotationTable, path: str | Path) -> None:
    """Save annotation table to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(annotations.to_list(), f, ensure_ascii=False)


def load_annotations(path: str | Path) -> AnnotationTable:
    """Load annotation table from a JSON file. Returns empty table if file doesn't exist."""
    path = Path(path)
    if not path.exists():
        return AnnotationTable()
    with open(path, "r", encoding="utf-8") as f:
        return AnnotationTable.from_list(json.load(f))


# ---- cache paths ----

def get_facets_cache_path(
    base_dir: str | Path,
    task_id: str,
    iteration: int,
) -> Path:
    """Cache path for the facet pool at a given iteration.

    Example:
        >>> get_facets_cache_path("cache", "sushi_v1", 2)
        Path('cache/sushi_v1/facets_iter2.json')
    """
    return Path(base_dir) / task_id / f"facets_iter{iteration}.json"


def get_annotations_cache_path(
    base_dir: str | Path,
    task_id: str,
    iteration: int,
) -> Path:
    """Cache path for annotations at a given iteration.

    Example:
        >>> get_annotations_cache_path("cache", "sushi_v1", 2)
        Path('cache/sushi_v1/annotations_iter2.json')
    """
    return Path(base_dir) / task_id / f"annotations_iter{iteration}.json"


# ---- skip helpers ----

def facets_needing_vocabulary(
    facets: List[Facet],
    cached_facets: List[Facet],
) -> List[Facet]:
    """Return facets that still need vocabulary built.

    A facet is skipped if a cached facet with the same id has non-empty vocabulary.
    """
    cached_with_vocab: Set[str] = {
        f.id for f in cached_facets if f.vocabulary
    }
    return [f for f in facets if f.id not in cached_with_vocab]


def facets_needing_annotation(
    facets: List[Facet],
    item_ids: List[str],
    annotations: AnnotationTable,
) -> List[Facet]:
    """Return facets that have unannotated items.

    A facet is skipped if every item in item_ids already has an entry
    in the annotation table for that facet.
    """
    item_id_set = set(item_ids)
    needed = []
    for facet in facets:
        annotated_items = set(annotations.by_facet(facet.id).keys())
        if not item_id_set.issubset(annotated_items):
            needed.append(facet)
    return needed
