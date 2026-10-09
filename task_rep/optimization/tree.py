"""
Refinement tree construction for task schema evaluation.

Builds a tree/DAG of schemas where edges represent refinement relationships
based on mutual information.

The algorithm:
1. Enumerate all facet subsets up to max_schema_facets, compute per-node metrics.
2. Greedy survival: process nodes by increasing size, reject if an
   already-accepted node captures the candidate (ref > tau at <= size).
3. Edge building: connect survivors where ref(A->B) > tau and size(A) < size(B).
4. Transitive reduction: remove shortcut edges to produce a clean tree.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, FrozenSet, List, Set, Tuple

from knowledge_items import KnowledgeBase
from facets import Facet, AnnotationTable

from .metrics import (
    compute_cells,
    compute_compactness,
    compute_discriminativeness,
    compute_eta_squared,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TreeNode:
    """A node in the refinement tree representing a schema (facet subset)."""

    id: str                # "|".join(sorted(facet_ids)), "" for empty schema
    facet_ids: List[str]   # sorted list of facet IDs
    size: int              # compactness = sum of non-na vocab labels
    D: float               # H(U|C), lower is better
    eta2: float            # 1 - eta_squared, lower is better
    H: float               # marginal entropy of cell-signature distribution

    @property
    def discriminativeness(self) -> float:
        """Alias for D (conditional entropy H(U|C), lower is better)."""
        return self.D

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "facet_ids": self.facet_ids,
            "size": self.size,
            "D": self.D,
            "eta2": self.eta2,
            "H": self.H,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> TreeNode:
        return TreeNode(
            id=data["id"],
            facet_ids=data["facet_ids"],
            size=data["size"],
            D=data["D"],
            eta2=data["eta2"],
            H=data["H"],
        )


@dataclass
class TreeEdge:
    """A directed refinement edge from parent (smaller) to child (larger)."""

    source_id: str              # parent node id
    target_id: str              # child node id
    MI: float                   # mutual information MI(source, target)
    refinement_score: float     # MI / H(source)
    marginal_D_gain: float      # D(source) - D(target), positive = target better
    marginal_eta2_gain: float   # eta2(source) - eta2(target), positive = target better
    added_facets: List[str]     # facet_ids in target but not source
    removed_facets: List[str]   # facet_ids in source but not target
    edge_type: str = "add"      # extensible

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "MI": self.MI,
            "refinement_score": self.refinement_score,
            "marginal_D_gain": self.marginal_D_gain,
            "marginal_eta2_gain": self.marginal_eta2_gain,
            "added_facets": self.added_facets,
            "removed_facets": self.removed_facets,
            "edge_type": self.edge_type,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> TreeEdge:
        return TreeEdge(
            source_id=data["source_id"],
            target_id=data["target_id"],
            MI=data["MI"],
            refinement_score=data["refinement_score"],
            marginal_D_gain=data["marginal_D_gain"],
            marginal_eta2_gain=data["marginal_eta2_gain"],
            added_facets=data["added_facets"],
            removed_facets=data["removed_facets"],
            edge_type=data.get("edge_type", "add"),
        )


@dataclass
class RefinementTree:
    """The full refinement tree: surviving nodes + edges after reduction."""

    nodes: List[TreeNode]
    edges: List[TreeEdge]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> RefinementTree:
        return RefinementTree(
            nodes=[TreeNode.from_dict(n) for n in data["nodes"]],
            edges=[TreeEdge.from_dict(e) for e in data["edges"]],
        )


@dataclass
class AcceptedSchemaSummary:
    """Prompt-facing summary of a schema that survived pruning.

    ``closest_*`` name the earlier-kept schema (no larger) with the highest
    overlap MI/H(this), which stayed at or below tau; empty if none.
    ``member_overlaps`` gives, for a multi-facet schema, H(facet)/H(this): how
    much of it each of its facets carries alone.
    """

    id: str
    facet_ids: List[str]
    size: int
    D: float
    eta2: float
    H: float
    closest_id: str = ""
    closest_facet_ids: List[str] = field(default_factory=list)
    closest_overlap: float = 0.0
    member_overlaps: Dict[str, float] = field(default_factory=dict)

    @staticmethod
    def from_node(node: TreeNode) -> "AcceptedSchemaSummary":
        return AcceptedSchemaSummary(
            id=node.id,
            facet_ids=list(node.facet_ids),
            size=node.size,
            D=node.D,
            eta2=node.eta2,
            H=node.H,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "facet_ids": self.facet_ids,
            "size": self.size,
            "D": self.D,
            "eta2": self.eta2,
            "H": self.H,
            "closest_id": self.closest_id,
            "closest_facet_ids": self.closest_facet_ids,
            "closest_overlap": self.closest_overlap,
            "member_overlaps": self.member_overlaps,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "AcceptedSchemaSummary":
        return AcceptedSchemaSummary(
            id=data["id"],
            facet_ids=data["facet_ids"],
            size=data["size"],
            D=data["D"],
            eta2=data["eta2"],
            H=data["H"],
            closest_id=data.get("closest_id", ""),
            closest_facet_ids=list(data.get("closest_facet_ids", [])),
            closest_overlap=data.get("closest_overlap", 0.0),
            member_overlaps=dict(data.get("member_overlaps", {})),
        )


@dataclass
class RejectedSchemaSummary:
    """Prompt-facing summary of a schema rejected by greedy pruning."""

    id: str
    facet_ids: List[str]
    size: int
    D: float
    eta2: float
    H: float
    rejected_by_id: str
    rejected_by_facet_ids: List[str]
    MI: float
    refinement_score: float
    refinement_threshold: float

    @staticmethod
    def from_nodes(
        rejected: TreeNode,
        rejected_by: TreeNode,
        MI: float,
        refinement_score: float,
        refinement_threshold: float,
    ) -> "RejectedSchemaSummary":
        return RejectedSchemaSummary(
            id=rejected.id,
            facet_ids=list(rejected.facet_ids),
            size=rejected.size,
            D=rejected.D,
            eta2=rejected.eta2,
            H=rejected.H,
            rejected_by_id=rejected_by.id,
            rejected_by_facet_ids=list(rejected_by.facet_ids),
            MI=MI,
            refinement_score=refinement_score,
            refinement_threshold=refinement_threshold,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "facet_ids": self.facet_ids,
            "size": self.size,
            "D": self.D,
            "eta2": self.eta2,
            "H": self.H,
            "rejected_by_id": self.rejected_by_id,
            "rejected_by_facet_ids": self.rejected_by_facet_ids,
            "MI": self.MI,
            "refinement_score": self.refinement_score,
            "refinement_threshold": self.refinement_threshold,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "RejectedSchemaSummary":
        return RejectedSchemaSummary(
            id=data["id"],
            facet_ids=data["facet_ids"],
            size=data["size"],
            D=data["D"],
            eta2=data["eta2"],
            H=data["H"],
            rejected_by_id=data["rejected_by_id"],
            rejected_by_facet_ids=data["rejected_by_facet_ids"],
            MI=data["MI"],
            refinement_score=data["refinement_score"],
            refinement_threshold=data["refinement_threshold"],
        )


@dataclass
class SchemaEvaluation:
    """Schema-pruning output split into graph and prompt-facing summaries."""

    tree: RefinementTree
    accepted_schemas: List[AcceptedSchemaSummary]
    rejected_schemas: List[RejectedSchemaSummary]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tree": self.tree.to_dict(),
            "accepted_schemas": [s.to_dict() for s in self.accepted_schemas],
            "rejected_schemas": [s.to_dict() for s in self.rejected_schemas],
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "SchemaEvaluation":
        return SchemaEvaluation(
            tree=RefinementTree.from_dict(data["tree"]),
            accepted_schemas=[
                AcceptedSchemaSummary.from_dict(s)
                for s in data["accepted_schemas"]
            ],
            rejected_schemas=[
                RejectedSchemaSummary.from_dict(s)
                for s in data["rejected_schemas"]
            ],
        )


# ---------------------------------------------------------------------------
# Entropy helpers
# ---------------------------------------------------------------------------

def compute_signature_entropy(
    cell_to_units: Dict[str, Dict[str, int]],
    base: int = 2,
) -> float:
    """
    Compute H(S) -- marginal entropy of the cell-signature distribution.

    Uses the same uniform-over-units weighting as compute_discriminativeness:
      P(C=c) = (1/N) * sum_u (m_{u,c} / m_{u,*})
      H(S)   = -sum_c P(C=c) * log(P(C=c))

    Args:
        cell_to_units: Mapping of cell signature -> {unit_id -> item_count}.
        base: Logarithm base.

    Returns:
        Marginal entropy H >= 0.  H=0 when there is only one cell.
    """
    unit_totals: Dict[str, int] = defaultdict(int)
    for unit_counts in cell_to_units.values():
        for unit_id, count in unit_counts.items():
            unit_totals[unit_id] += count

    n_units = len(unit_totals)
    if n_units == 0:
        return 0.0

    # Compute P(C=c) for each cell, then entropy
    h = 0.0
    for unit_counts in cell_to_units.values():
        weights = {}
        for unit_id, count in unit_counts.items():
            if count > 0:
                weights[unit_id] = count / unit_totals[unit_id]

        w_sum = sum(weights.values())
        if w_sum == 0:
            continue

        p_c = w_sum / n_units
        if p_c > 0:
            h -= p_c * math.log(p_c, base)

    return h


def compute_node_mi(H_a: float, H_b: float, H_union: float) -> float:
    """
    Compute mutual information MI(A, B) = H(A) + H(B) - H(A union B).

    Clamps to 0 to handle floating-point noise.
    """
    return max(0.0, H_a + H_b - H_union)


# ---------------------------------------------------------------------------
# Node quality helper
# ---------------------------------------------------------------------------

def node_quality(node: TreeNode, metric_mode: str) -> float:
    """
    Quality value of a node (higher = better).

    Both D and eta2 are stored as lower-is-better, so we negate.
    """
    if metric_mode == "entropy":
        return -node.D
    else:
        return -node.eta2


# ---------------------------------------------------------------------------
# Build tree nodes
# ---------------------------------------------------------------------------

def build_tree_nodes(
    knowledge_base: KnowledgeBase,
    facets: List[Facet],
    annotations: AnnotationTable,
    max_schema_facets: int = 3,
    metric_mode: str = "entropy",
    eta_squared_weighting: str = "equal",
) -> Tuple[List[TreeNode], Dict[FrozenSet[str], float]]:
    """
    Enumerate all facet subsets and compute per-node metrics.

    Brute-force enumeration of subsets up to ``max_schema_facets``; computes
    D, eta2, and H for each subset.

    Args:
        knowledge_base: Items with unit_ids assigned.
        facets: Pool of candidate facets.
        annotations: Current label assignments.
        max_schema_facets: Maximum facets per schema.
        metric_mode: "entropy" | "coherence_ordinal" | "coherence_temporal"
        eta_squared_weighting: "equal" | "weighted"

    Returns:
        Tuple of (list of TreeNodes, memoized H values keyed by frozenset of facet_ids).
    """
    # Memoize cells and entropy by facet id set
    h_memo: Dict[FrozenSet[str], float] = {}

    def _get_cells(subset_facets: List[Facet]) -> Dict[str, Dict[str, int]]:
        facet_ids = [f.id for f in subset_facets]
        sub_ann = annotations.filter_facets(facet_ids)
        return compute_cells(knowledge_base, subset_facets, sub_ann)

    def _get_H(subset_facets: List[Facet]) -> float:
        key = frozenset(f.id for f in subset_facets)
        if key in h_memo:
            return h_memo[key]
        cells = _get_cells(subset_facets)
        h = compute_signature_entropy(cells)
        h_memo[key] = h
        return h

    if not facets:
        cells = compute_cells(knowledge_base, [], annotations)
        d = compute_discriminativeness(cells)
        h = compute_signature_entropy(cells)
        h_memo[frozenset()] = h
        node = TreeNode(
            id="",
            facet_ids=[],
            size=0,
            D=d,
            eta2=1.0,  # worst
            H=h,
        )
        return [node], h_memo

    max_size = min(max_schema_facets, len(facets))
    nodes: List[TreeNode] = []

    for size in range(max_size + 1):
        for combo in combinations(range(len(facets)), size):
            subset_facets = [facets[i] for i in combo]
            facet_ids = sorted(f.id for f in subset_facets)
            node_id = "|".join(facet_ids)

            sub_ann = annotations.filter_facets(facet_ids)
            cells = compute_cells(knowledge_base, subset_facets, sub_ann)

            d = compute_discriminativeness(cells)

            pos_mode = "ordinal" if metric_mode == "coherence_ordinal" else "time"
            eta2 = compute_eta_squared(
                knowledge_base, subset_facets, sub_ann,
                position_mode=pos_mode,
                weighting=eta_squared_weighting,
            )

            h = compute_signature_entropy(cells)
            key = frozenset(facet_ids)
            h_memo[key] = h

            c = compute_compactness(subset_facets)

            nodes.append(TreeNode(
                id=node_id,
                facet_ids=facet_ids,
                size=c,
                D=d,
                eta2=eta2,
                H=h,
            ))

    return nodes, h_memo


# ---------------------------------------------------------------------------
# Prune and connect
# ---------------------------------------------------------------------------

def prune_and_connect(
    nodes: List[TreeNode],
    knowledge_base: KnowledgeBase,
    facets_by_id: Dict[str, Facet],
    annotations: AnnotationTable,
    h_memo: Dict[FrozenSet[str], float],
    refinement_threshold: float = 0.7,
    metric_mode: str = "entropy",
) -> RefinementTree:
    """
    Build a refinement tree via greedy pruning + edge building + transitive reduction.

    Phase A: Greedy survival -- process nodes by increasing size (ties broken
    by decreasing quality, then id). Reject a candidate if an already-accepted
    keeper captures it (MI/H(candidate) > tau at <= size).

    Phase B: Edge building -- for every pair of survivors, add directed edge
    A->B if ref(A->B) = MI(A,B)/H(A) > tau and A.size < B.size.

    Phase C: Transitive reduction -- remove shortcut edges reachable via
    longer paths.

    Args:
        nodes: All enumerated tree nodes.
        knowledge_base: For computing MI of unions not in h_memo.
        facets_by_id: Facet lookup.
        annotations: Label assignments.
        h_memo: Pre-computed H values from build_tree_nodes.
        refinement_threshold: Tau threshold.
        metric_mode: Quality metric for sorting.

    Returns:
        RefinementTree with surviving nodes and reduced edges.
    """
    return evaluate_schemas(
        nodes, knowledge_base, facets_by_id, annotations, h_memo,
        refinement_threshold=refinement_threshold,
        metric_mode=metric_mode,
    ).tree


def evaluate_schemas(
    nodes: List[TreeNode],
    knowledge_base: KnowledgeBase,
    facets_by_id: Dict[str, Facet],
    annotations: AnnotationTable,
    h_memo: Dict[FrozenSet[str], float],
    refinement_threshold: float = 0.7,
    metric_mode: str = "entropy",
) -> SchemaEvaluation:
    """
    Build schema evaluation output for optimization and downstream prompts.

    The returned RefinementTree contains only accepted schemas and graph edges.
    Rejected schemas are returned separately with the accepted schema that
    rejected them, so prompt code can consume schema summaries without taking a
    dependency on tree traversal.
    """
    tau = refinement_threshold

    def _get_H_union(node_a: TreeNode, node_b: TreeNode) -> float:
        """Get H of the union of two nodes' facet sets."""
        return _get_H_union_ids(frozenset(node_a.facet_ids) | frozenset(node_b.facet_ids))

    def _get_H_union_ids(union_ids: FrozenSet[str]) -> float:
        if union_ids in h_memo:
            return h_memo[union_ids]
        # Compute on the fly
        union_facets = [facets_by_id[fid] for fid in union_ids if fid in facets_by_id]
        sub_ann = annotations.filter_facets(list(union_ids))
        cells = compute_cells(knowledge_base, union_facets, sub_ann)
        h = compute_signature_entropy(cells)
        h_memo[union_ids] = h
        return h

    def _compute_mi(a: TreeNode, b: TreeNode) -> float:
        h_union = _get_H_union(a, b)
        return compute_node_mi(a.H, b.H, h_union)

    # Sort: ascending size, descending quality, ascending id
    sorted_nodes = sorted(nodes, key=lambda n: (
        n.size,
        -node_quality(n, metric_mode),
        n.id,
    ))

    # Phase A: greedy survival
    accepted: List[TreeNode] = []
    rejected: List[RejectedSchemaSummary] = []
    closest: Dict[str, Tuple[TreeNode, float]] = {}
    for candidate in sorted_nodes:
        dominated = False
        for keeper in accepted:
            mi = _compute_mi(keeper, candidate)
            ref_from_candidate = mi / candidate.H if candidate.H > 0 else 0.0
            if keeper.size <= candidate.size and keeper.facet_ids and (
                candidate.id not in closest or ref_from_candidate > closest[candidate.id][1]
            ):
                closest[candidate.id] = (keeper, ref_from_candidate)
            if ref_from_candidate > tau and keeper.size <= candidate.size:
                rejected.append(RejectedSchemaSummary.from_nodes(
                    rejected=candidate,
                    rejected_by=keeper,
                    MI=mi,
                    refinement_score=ref_from_candidate,
                    refinement_threshold=tau,
                ))
                dominated = True
                break
        if not dominated:
            accepted.append(candidate)

    # Phase B: edge building among survivors
    edges: List[TreeEdge] = []
    for i in range(len(accepted)):
        for j in range(i + 1, len(accepted)):
            a = accepted[i]
            b = accepted[j]
            mi = _compute_mi(a, b)

            # Check A -> B direction
            ref_ab = mi / a.H if a.H > 0 else 0.0
            if ref_ab > tau and a.size < b.size:
                a_set = set(a.facet_ids)
                b_set = set(b.facet_ids)
                edges.append(TreeEdge(
                    source_id=a.id,
                    target_id=b.id,
                    MI=mi,
                    refinement_score=ref_ab,
                    marginal_D_gain=a.D - b.D,
                    marginal_eta2_gain=a.eta2 - b.eta2,
                    added_facets=sorted(b_set - a_set),
                    removed_facets=sorted(a_set - b_set),
                    edge_type="add",
                ))

            # Check B -> A direction
            ref_ba = mi / b.H if b.H > 0 else 0.0
            if ref_ba > tau and b.size < a.size:
                a_set = set(a.facet_ids)
                b_set = set(b.facet_ids)
                edges.append(TreeEdge(
                    source_id=b.id,
                    target_id=a.id,
                    MI=mi,
                    refinement_score=ref_ba,
                    marginal_D_gain=b.D - a.D,
                    marginal_eta2_gain=b.eta2 - a.eta2,
                    added_facets=sorted(a_set - b_set),
                    removed_facets=sorted(b_set - a_set),
                    edge_type="add",
                ))

    # Phase C: transitive reduction
    edges = _transitive_reduction(accepted, edges)

    tree = RefinementTree(nodes=accepted, edges=edges)
    summaries = []
    for node in accepted:
        summary = AcceptedSchemaSummary.from_node(node)
        if node.id in closest:
            keeper, overlap = closest[node.id]
            summary.closest_id = keeper.id
            summary.closest_facet_ids = list(keeper.facet_ids)
            summary.closest_overlap = overlap
        if len(node.facet_ids) > 1 and node.H > 0:
            # MI(single, superset) = H(single)
            summary.member_overlaps = {
                fid: _get_H_union_ids(frozenset([fid])) / node.H for fid in node.facet_ids
            }
        summaries.append(summary)
    return SchemaEvaluation(tree=tree, accepted_schemas=summaries, rejected_schemas=rejected)


def _transitive_reduction(
    nodes: List[TreeNode],
    edges: List[TreeEdge],
) -> List[TreeEdge]:
    """
    Remove redundant edges reachable via longer paths (BFS-based).

    An edge A->C is removed if there exists a path A->B->...->C of length >= 2.
    """
    # Build adjacency list
    adj: Dict[str, Set[str]] = defaultdict(set)
    for e in edges:
        adj[e.source_id].add(e.target_id)

    def _is_reachable_without_direct(source: str, target: str) -> bool:
        """BFS from source's other children to see if target is reachable."""
        children = adj.get(source, set())
        visited: Set[str] = set()
        queue: List[str] = [c for c in children if c != target]
        while queue:
            cur = queue.pop(0)
            if cur == target:
                return True
            if cur in visited:
                continue
            visited.add(cur)
            for nxt in adj.get(cur, set()):
                if nxt not in visited:
                    queue.append(nxt)
        return False

    return [e for e in edges if not _is_reachable_without_direct(e.source_id, e.target_id)]


# ---------------------------------------------------------------------------
# Improvement detection
# ---------------------------------------------------------------------------

def compare_tree_quality(tree_a: RefinementTree, tree_b: RefinementTree) -> int:
    """
    Compare two trees by quality.

    Returns 1 if tree_a is better, -1 if tree_b is better, 0 if equal.

    Primary criterion: more surviving nodes is better.
    Tiebreaker: fewer unique facet IDs across all nodes is better.
    """
    nodes_a = len(tree_a.nodes)
    nodes_b = len(tree_b.nodes)
    if nodes_a != nodes_b:
        return 1 if nodes_a > nodes_b else -1

    unique_a = len({fid for node in tree_a.nodes for fid in node.facet_ids})
    unique_b = len({fid for node in tree_b.nodes for fid in node.facet_ids})
    if unique_a != unique_b:
        return 1 if unique_a < unique_b else -1

    return 0


def compute_tree_depth(tree: RefinementTree) -> int:
    """
    Max path length from any root to any leaf in the refinement tree.

    Returns 0 if the tree has no edges (all nodes are roots/leaves).
    """
    if not tree.edges:
        return 0
    children: Dict[str, List[str]] = defaultdict(list)
    for e in tree.edges:
        children[e.source_id].append(e.target_id)
    parents = {e.target_id for e in tree.edges}
    roots = [n.id for n in tree.nodes if n.id not in parents]
    max_depth = 0
    for root in roots:
        stack = [(root, 0)]
        while stack:
            nid, d = stack.pop()
            max_depth = max(max_depth, d)
            for child in children.get(nid, []):
                stack.append((child, d + 1))
    return max_depth
