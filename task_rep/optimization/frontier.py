"""
Pareto frontier computation for task schema evaluation.

Port of server/src/framework_split.py:52-133 (compute_frontier_knapsack)
and the frontier comparison/sampling logic.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, FrozenSet, List, Tuple

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
    compute_quality_metric,
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
    max_schema_facets: int = 4,
    metric_mode: str = "entropy",
    eta_squared_weighting: str = "equal",
) -> List[FrontierPoint]:
    """
    Compute the Pareto frontier of schemas by brute-force enumeration.

    Enumerates all subsets of up to MAX_SCHEMA_FACETS facets. For each
    achievable compactness value, the subset with the lowest quality score
    is retained. Dominated points are then removed.

    The quality metric is not additively separable across facets (facet
    interactions can be large), so knapsack DP cannot guarantee optimality.
    Brute-force over subsets of bounded size is correct and efficient in
    practice (n^K evaluations, K=MAX_SCHEMA_FACETS).

    Args:
        knowledge_base: Items with unit_ids assigned.
        facets: Pool of candidate facets.
        annotations: Current label assignments.
        metric_mode: "entropy" | "coherence_ordinal" | "coherence_temporal"
        eta_squared_weighting: "equal" | "weighted" (only for coherence modes)

    Returns:
        List of FrontierPoint sorted by ascending compactness (and thus
        strictly decreasing discriminativeness/quality score).
    """
    num_units = count_unique_units(knowledge_base)

    # Memoized evaluation: indices_tuple -> quality score (lower is better)
    memo: Dict[Tuple[int, ...], float] = {}

    def eval_subset(indices: Tuple[int, ...]) -> float:
        if indices in memo:
            return memo[indices]
        subset_facets = [facets[i] for i in indices]
        facet_ids = [f.id for f in subset_facets]
        subset_annotations = annotations.filter_facets(facet_ids)
        d = compute_quality_metric(
            knowledge_base, subset_facets, subset_annotations,
            metric_mode=metric_mode,
            eta_squared_weighting=eta_squared_weighting,
        )
        memo[indices] = d
        return d

    if not facets:
        d = eval_subset(())
        return [FrontierPoint(discriminativeness=d, compactness=0, facet_ids=[])]

    weights = [compute_compactness([f]) for f in facets]
    max_size = min(max_schema_facets, len(facets))

    # best[compactness] = (min_D, winning_subset)
    best: Dict[int, Tuple[float, Tuple[int, ...]]] = {}
    # Index sets of subsets that achieve D=0; all strict supersets are dominated
    # (adding facets can only keep D at 0, never improve it, and costs more C).
    perfect: List[FrozenSet[int]] = []

    for size in range(max_size + 1):
        for combo in combinations(range(len(facets)), size):
            # Monotonicity pruning: a superset of a D=0 subset achieves D=0 at
            # higher compactness — strictly dominated, skip.
            if any(p.issubset(combo) for p in perfect):
                continue
            c = sum(weights[i] for i in combo)
            d = eval_subset(combo)
            if c not in best or d < best[c][0]:
                best[c] = (d, combo)
            if d == 0.0:
                perfect.append(frozenset(combo))

    # Collect all (compactness, D) points sorted by ascending compactness
    raw: List[FrontierPoint] = sorted(
        [
            FrontierPoint(
                discriminativeness=d,
                compactness=c,
                facet_ids=[facets[i].id for i in subset],
            )
            for c, (d, subset) in best.items()
        ],
        key=lambda p: p.compactness,
    )

    # Remove dominated points: keep only points where D strictly decreases
    # as compactness increases.
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

    Both frontiers are sorted by ascending compactness and assumes that the new_frontier is at least no worse than the old_frontier.
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

    # Old exhausted but new has remaining points at higher C.
    # If new's minimum D beats old's minimum D, that's a real improvement.
    if old_i == len(old_clipped) and new_i < len(new_clipped):
        if new_clipped[-1].discriminativeness < old_clipped[-1].discriminativeness:
            return True

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
