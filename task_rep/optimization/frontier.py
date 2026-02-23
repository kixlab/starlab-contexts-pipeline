"""
Pareto frontier computation for task schema evaluation.

Port of server/src/framework_split.py:52-133 (compute_frontier_knapsack)
and the frontier comparison/sampling logic.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from knowledge_items import (
    KnowledgeBase,
)
from facets import (
    Facet,
    AnnotationTable,
)

from .metrics import (
    compute_cells,
    compute_compactness,
    compute_discriminativeness,
    count_unique_units,
)


@dataclass
class FrontierPoint:
    """A single point on the Pareto frontier.

    Represents a schema (subset of facets) with its evaluation scores.
    Points are sorted by ascending compactness on the frontier, which
    means discriminativeness is strictly decreasing.
    """

    discriminativeness: float
    compactness: int
    facet_ids: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "discriminativeness": self.discriminativeness,
            "compactness": self.compactness,
            "facet_ids": self.facet_ids,
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> FrontierPoint:
        return FrontierPoint(
            discriminativeness=data["discriminativeness"],
            compactness=data["compactness"],
            facet_ids=data["facet_ids"],
        )


def compute_frontier(
    knowledge_base: KnowledgeBase,
    facets: List[Facet],
    annotations: AnnotationTable,
) -> List[FrontierPoint]:
    """
    Compute the Pareto frontier of schemas via 0/1 knapsack DP.

    Each facet is a knapsack item with weight = vocabulary size (non-NA labels).
    For each achievable total weight (compactness), the DP finds the facet subset
    that minimizes discriminativeness. Dominated points are then removed.

    Port of server's compute_frontier_knapsack() (framework_split.py:52-133).

    Args:
        knowledge_base: Items with unit_ids assigned.
        facets: Pool of candidate facets.
        annotations: Current label assignments.

    Returns:
        List of FrontierPoint sorted by ascending compactness (and thus
        strictly decreasing discriminativeness).
    """
    num_units = count_unique_units(knowledge_base)

    # Memoized evaluation: indices_tuple -> discriminativeness
    memo: Dict[Tuple[int, ...], float] = {}

    def eval_subset(indices: Tuple[int, ...]) -> float:
        if indices in memo:
            return memo[indices]
        subset_facets = [facets[i] for i in indices]
        facet_ids = [f.id for f in subset_facets]
        subset_annotations = annotations.filter_facets(facet_ids)
        cells = compute_cells(knowledge_base, subset_facets, subset_annotations)
        d = compute_discriminativeness(cells, num_units)
        memo[indices] = d
        return d

    if not facets:
        d = eval_subset(())
        return [FrontierPoint(discriminativeness=d, compactness=0, facet_ids=[])]

    weights = [compute_compactness([f]) for f in facets]
    max_weight = sum(w for w in weights if w > 0)

    if max_weight == 0:
        d = eval_subset(())
        return [FrontierPoint(discriminativeness=d, compactness=0, facet_ids=[])]

    # DP state: for each achievable weight w, track min D and the achieving subset
    dp_d: List[float] = [float("inf")] * (max_weight + 1)
    dp_subset: List[Optional[Tuple[int, ...]]] = [None] * (max_weight + 1)

    # Base case: empty subset
    dp_d[0] = eval_subset(())
    dp_subset[0] = ()

    # 0/1 knapsack — iterate weights backwards to prevent reuse
    for i, w_i in enumerate(weights):
        if w_i <= 0:
            continue
        for w in range(max_weight - w_i, -1, -1):
            if dp_subset[w] is None:
                continue
            new_w = w + w_i
            new_subset = tuple(sorted(dp_subset[w] + (i,)))
            new_d = eval_subset(new_subset)
            if new_d < dp_d[new_w]:
                dp_d[new_w] = new_d
                dp_subset[new_w] = new_subset

    # Collect all achievable (compactness, D) points
    raw: List[FrontierPoint] = []
    for w in range(max_weight + 1):
        if dp_subset[w] is None:
            continue
        raw.append(FrontierPoint(
            discriminativeness=dp_d[w],
            compactness=w,
            facet_ids=[facets[i].id for i in dp_subset[w]],
        ))

    # Smooth: remove dominated points.
    # Raw is sorted by ascending compactness. Keep only points where D is
    # strictly less than the previous kept point (so D decreases as C increases).
    smoothed: List[FrontierPoint] = []
    for point in raw:
        if smoothed and smoothed[-1].discriminativeness <= point.discriminativeness:
            continue
        smoothed.append(point)
    return smoothed


def clip_frontier(
    frontier: List[FrontierPoint],
    min_d: float = 0.01,
) -> List[FrontierPoint]:
    """
    Remove frontier points that are already near-perfectly discriminating.

    Points with D ≤ min_d represent schemas that already separate units
    well enough — no need to sample them for further expansion.
    Near-duplicate D values are also deduplicated.

    Port of server's clip_frontier logic (framework_split.py:170-178).
    """
    clipped = []
    for point in frontier:
        if point.discriminativeness <= min_d:
            continue
        if clipped and math.isclose(clipped[-1].discriminativeness, point.discriminativeness):
            continue
        clipped.append(point)
    return clipped


def is_pareto_improved(
    old_frontier: List[FrontierPoint],
    new_frontier: List[FrontierPoint],
    min_d: float = 0.01,
) -> bool:
    """
    Check whether the new frontier meaningfully improves on the old one.

    Compares the clipped versions of both frontiers to avoid spurious
    improvements from merely extending the compactness range. A new
    point added far into high-compactness territory (many facets) would
    otherwise always trigger improvement even with no real gain.

    Returns True if any clipped new point dominates a clipped old point:
    - new_C ≤ old_C and new_D < old_D, OR
    - new_C < old_C and new_D ≤ old_D.

    Both frontiers are sorted by ascending compactness.
    """
    old_clipped = clip_frontier(old_frontier, min_d)
    new_clipped = clip_frontier(new_frontier, min_d)

    if not old_clipped:
        return False
    if not new_clipped:
        return False

    old_i, new_i = 0, 0
    while old_i < len(old_clipped) and new_i < len(new_clipped):
        old_c = old_clipped[old_i].compactness
        old_d = old_clipped[old_i].discriminativeness
        new_c = new_clipped[new_i].compactness
        new_d = new_clipped[new_i].discriminativeness

        if new_c <= old_c and new_d < old_d:
            return True
        if new_c < old_c and new_d <= old_d:
            return True

        if old_c <= new_c:
            old_i += 1
        else:
            new_i += 1

    return False


def sample_schemas(
    frontier: List[FrontierPoint],
    facets_by_id: Dict[str, Facet],
    k: int = 2,
) -> List[Tuple[List[Facet], FrontierPoint]]:
    """
    Sample k schemas from the Pareto frontier.

    Randomly samples from the (already clipped) frontier, resolving
    facet_ids back to Facet objects.

    Args:
        frontier: Clipped frontier (output of clip_frontier).
        facets_by_id: Mapping of facet_id -> Facet for resolution.
        k: Number of schemas to sample.

    Returns:
        List of (facet_list, frontier_point) tuples.
    """
    if not frontier:
        return []

    sampled = random.sample(frontier, min(k, len(frontier)))

    return [
        (
            [facets_by_id[fid] for fid in point.facet_ids if fid in facets_by_id],
            point,
        )
        for point in sampled
    ]
