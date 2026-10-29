"""
Unit tests for the refinement tree module.

Tests signature entropy, mutual information, tree node building,
pruning/connecting, transitive reduction, sampling, and improvement detection.

Run with: python -m pytest tests/test_tree.py -v
"""

from __future__ import annotations

import math
import pytest

from knowledge_items.schema import (
    KnowledgeBase,
    KnowledgeItem,
    KnowledgeItemType,
    KnowledgeSource,
)
from facets.schema import AnnotationTable, Facet, FacetLabel
from optimization.tree import (
    TreeNode,
    TreeEdge,
    RefinementTree,
    SchemaEvaluation,
    build_tree_nodes,
    compute_node_mi,
    compute_signature_entropy,
    evaluate_schemas,
    prune_and_connect,
    compare_tree_quality,
    _transitive_reduction,
)
from optimization.metrics import compute_cells


# ============= Helpers =============

def make_source(video_id: str = "v1") -> KnowledgeSource:
    return KnowledgeSource(video_id=video_id)


def make_item(
    id: str,
    unit_id: str | None = None,
    video_id: str = "v1",
) -> KnowledgeItem:
    return KnowledgeItem(
        id=id,
        type=KnowledgeItemType.ATOMIC,
        text=f"Item {id}",
        source=make_source(video_id),
        unit_id=unit_id,
    )


def make_facet(id: str, title: str, labels: list[str]) -> Facet:
    return Facet(
        id=id,
        title=title,
        title_plural=title + "s",
        scope="global",
        definition=f"Definition of {title}",
        justification="Testing",
        guidelines=[],
        vocabulary=[FacetLabel(label=lbl, definition=f"Def {lbl}") for lbl in labels],
    )


def make_tree_node(
    facet_ids: list[str],
    size: int = 0,
    D: float = 1.0,
    eta2: float = 1.0,
    H: float = 0.0,
) -> TreeNode:
    return TreeNode(
        id="|".join(sorted(facet_ids)),
        facet_ids=sorted(facet_ids),
        size=size,
        D=D,
        eta2=eta2,
        H=H,
    )


# ============= Shared fixtures =============

@pytest.fixture
def three_unit_kb():
    """3 units (u1, u2, u3), one item each."""
    return KnowledgeBase(items=[
        make_item("item1", unit_id="u1"),
        make_item("item2", unit_id="u2"),
        make_item("item3", unit_id="u3"),
    ])


@pytest.fixture
def f1():
    """Facet 'Difficulty' with labels [a, b]. Compactness = 2."""
    return make_facet("f1", "Difficulty", ["a", "b"])


@pytest.fixture
def f2():
    """Facet 'Tool' with labels [x, y, z]. Compactness = 3."""
    return make_facet("f2", "Tool", ["x", "y", "z"])


@pytest.fixture
def f3():
    """Facet 'Level' with labels [lo, hi]. Compactness = 2."""
    return make_facet("f3", "Level", ["lo", "hi"])


@pytest.fixture
def annotations_f1_f2():
    """
    item1: f1=a, f2=x  (unit u1)
    item2: f1=b, f2=y  (unit u2)
    item3: f1=b, f2=z  (unit u3)

    f1 alone: 2 cells, one has u2+u3 mixed
    f2 alone: 3 cells, perfect separation
    f1+f2: 3 cells, perfect separation
    """
    ann = AnnotationTable()
    ann[("item1", "f1")] = "a"
    ann[("item2", "f1")] = "b"
    ann[("item3", "f1")] = "b"
    ann[("item1", "f2")] = "x"
    ann[("item2", "f2")] = "y"
    ann[("item3", "f2")] = "z"
    return ann


@pytest.fixture
def annotations_f1_f2_f3():
    """f1, f2, f3 annotations where f3 duplicates f1 exactly."""
    ann = AnnotationTable()
    ann[("item1", "f1")] = "a"
    ann[("item2", "f1")] = "b"
    ann[("item3", "f1")] = "b"
    ann[("item1", "f2")] = "x"
    ann[("item2", "f2")] = "y"
    ann[("item3", "f2")] = "z"
    ann[("item1", "f3")] = "lo"
    ann[("item2", "f3")] = "hi"
    ann[("item3", "f3")] = "hi"
    return ann


# --- Independent facets fixtures (6 items, 2x3 orthogonal grid) ---

@pytest.fixture
def six_unit_kb():
    """6 items, 6 units, from 2 videos."""
    return KnowledgeBase(items=[
        make_item("i1", unit_id="u1", video_id="v1"),
        make_item("i2", unit_id="u2", video_id="v1"),
        make_item("i3", unit_id="u3", video_id="v1"),
        make_item("i4", unit_id="u4", video_id="v2"),
        make_item("i5", unit_id="u5", video_id="v2"),
        make_item("i6", unit_id="u6", video_id="v2"),
    ])


@pytest.fixture
def f_temporal():
    """Facet partitioning by video: early vs late. Compactness = 2."""
    return make_facet("f_temporal", "Temporal", ["early", "late"])


@pytest.fixture
def f_content():
    """Facet partitioning by content type: A/B/C. Compactness = 3."""
    return make_facet("f_content", "Content", ["typeA", "typeB", "typeC"])


@pytest.fixture
def independent_annotations():
    """
    f_temporal: {u1,u2,u3}=early, {u4,u5,u6}=late  (by video)
    f_content:  {u1,u4}=typeA, {u2,u5}=typeB, {u3,u6}=typeC (orthogonal)
    MI(f_temporal, f_content) = 0 because knowing one tells you nothing
    about the other.
    """
    ann = AnnotationTable()
    for item_id, f_t, f_c in [
        ("i1", "early", "typeA"),
        ("i2", "early", "typeB"),
        ("i3", "early", "typeC"),
        ("i4", "late", "typeA"),
        ("i5", "late", "typeB"),
        ("i6", "late", "typeC"),
    ]:
        ann[(item_id, "f_temporal")] = f_t
        ann[(item_id, "f_content")] = f_c
    return ann


# ============= Tests: compute_signature_entropy =============

class TestSignatureEntropy:
    def test_empty_cells(self):
        """No cells → H = 0."""
        assert compute_signature_entropy({}) == 0.0

    def test_single_cell(self, three_unit_kb, f1, annotations_f1_f2):
        """All items in one cell → H = 0 (only one signature)."""
        # No facets → everything in one cell
        cells = compute_cells(three_unit_kb, [], AnnotationTable())
        h = compute_signature_entropy(cells)
        assert h == 0.0

    def test_perfect_separation_f2(self, three_unit_kb, f2, annotations_f1_f2):
        """3 cells with one unit each → H = log2(3)."""
        cells = compute_cells(
            three_unit_kb, [f2], annotations_f1_f2.filter_facets(["f2"])
        )
        h = compute_signature_entropy(cells)
        # Each cell has exactly one unit with weight 1.0
        # P(c) = (1/3)*1.0 for each of 3 cells → H = log2(3)
        assert math.isclose(h, math.log2(3), rel_tol=1e-9)

    def test_partial_separation_f1(self, three_unit_kb, f1, annotations_f1_f2):
        """
        f1 produces 2 cells:
          cell "a": u1 only (weight 1.0)
          cell "b": u2 and u3 (each weight 1.0)
        P(a) = (1/3)*1 = 1/3
        P(b) = (1/3)*(1+1) = 2/3
        H = -(1/3)*log2(1/3) - (2/3)*log2(2/3)
        """
        cells = compute_cells(
            three_unit_kb, [f1], annotations_f1_f2.filter_facets(["f1"])
        )
        h = compute_signature_entropy(cells)
        expected = -(1 / 3) * math.log2(1 / 3) - (2 / 3) * math.log2(2 / 3)
        assert math.isclose(h, expected, rel_tol=1e-9)

    def test_nonnegative(self, three_unit_kb, f1, f2, annotations_f1_f2):
        """H should always be >= 0."""
        for facets in [[], [f1], [f2], [f1, f2]]:
            fids = [f.id for f in facets]
            cells = compute_cells(
                three_unit_kb, facets, annotations_f1_f2.filter_facets(fids)
            )
            assert compute_signature_entropy(cells) >= 0.0


# ============= Tests: compute_node_mi =============

class TestNodeMI:
    def test_identical_schemas(self, three_unit_kb, f1, annotations_f1_f2):
        """MI(A, A) = H(A)."""
        cells = compute_cells(
            three_unit_kb, [f1], annotations_f1_f2.filter_facets(["f1"])
        )
        h_a = compute_signature_entropy(cells)
        # Union of A and A is A, so H_union = H_a
        mi = compute_node_mi(h_a, h_a, h_a)
        assert math.isclose(mi, h_a, rel_tol=1e-9)

    def test_nonnegative(self):
        """MI should never be negative."""
        # Even with floating point noise, clamp ensures >= 0
        assert compute_node_mi(1.0, 1.0, 2.5) == 0.0  # would be -0.5 without clamp

    def test_subset_relation(self, three_unit_kb, f1, f2, annotations_f1_f2):
        """
        If A's signature is a coarsening of B's (A ⊂ B as facet sets),
        then H(A∪B) = H(B), so MI(A,B) = H(A) + H(B) - H(B) = H(A).
        """
        cells_f1 = compute_cells(
            three_unit_kb, [f1], annotations_f1_f2.filter_facets(["f1"])
        )
        cells_f2 = compute_cells(
            three_unit_kb, [f2], annotations_f1_f2.filter_facets(["f2"])
        )
        cells_union = compute_cells(
            three_unit_kb, [f1, f2], annotations_f1_f2.filter_facets(["f1", "f2"])
        )
        h_a = compute_signature_entropy(cells_f1)
        h_b = compute_signature_entropy(cells_f2)
        h_union = compute_signature_entropy(cells_union)

        mi = compute_node_mi(h_a, h_b, h_union)
        # In this specific case: f2 perfectly separates all 3 units,
        # so f1+f2 has the same 3 cells as f2. H(union) = H(f2).
        # MI = H(f1) + H(f2) - H(f2) = H(f1)
        assert math.isclose(h_union, h_b, rel_tol=1e-9)
        assert math.isclose(mi, h_a, rel_tol=1e-9)


# ============= Tests: build_tree_nodes =============

class TestBuildTreeNodes:
    def test_no_facets(self, three_unit_kb):
        """With no facets, produces a single empty-schema node."""
        nodes, h_memo = build_tree_nodes(
            three_unit_kb, [], AnnotationTable(), max_schema_facets=3
        )
        assert len(nodes) == 1
        assert nodes[0].id == ""
        assert nodes[0].facet_ids == []
        assert nodes[0].size == 0

    def test_node_count(self, three_unit_kb, f1, f2, annotations_f1_f2):
        """With 2 facets and max_schema_facets=2: C(2,0)+C(2,1)+C(2,2) = 4 nodes."""
        nodes, h_memo = build_tree_nodes(
            three_unit_kb, [f1, f2], annotations_f1_f2, max_schema_facets=2
        )
        assert len(nodes) == 4  # {}, {f1}, {f2}, {f1,f2}

    def test_h_memo_populated(self, three_unit_kb, f1, f2, annotations_f1_f2):
        """h_memo should have H values for all enumerated subsets."""
        nodes, h_memo = build_tree_nodes(
            three_unit_kb, [f1, f2], annotations_f1_f2, max_schema_facets=2
        )
        assert frozenset() in h_memo
        assert frozenset(["f1"]) in h_memo
        assert frozenset(["f2"]) in h_memo
        assert frozenset(["f1", "f2"]) in h_memo

    def test_perfect_separator_has_D_zero(self, three_unit_kb, f2, annotations_f1_f2):
        """f2 perfectly separates 3 units → D = 0."""
        nodes, _ = build_tree_nodes(
            three_unit_kb, [f2], annotations_f1_f2, max_schema_facets=1
        )
        f2_node = [n for n in nodes if n.facet_ids == ["f2"]][0]
        assert f2_node.D == 0.0

    def test_node_ids_deterministic(self, three_unit_kb, f1, f2, annotations_f1_f2):
        """Node ids should be sorted and joined."""
        nodes, _ = build_tree_nodes(
            three_unit_kb, [f1, f2], annotations_f1_f2, max_schema_facets=2
        )
        ids = {n.id for n in nodes}
        assert "" in ids
        assert "f1" in ids
        assert "f2" in ids
        assert "f1|f2" in ids


# ============= Tests: prune_and_connect =============

class TestPruneAndConnect:
    def test_redundant_node_pruned(self, three_unit_kb, f1, f2, annotations_f1_f2):
        """
        With tau=0.3, f1 (size=2) captures enough of f2 (size=3) that f2
        gets pruned (f1 is smaller and accepted first). f1|f2 also gets
        pruned since f1 captures it too. Only "" and f1 survive.
        """
        nodes, h_memo = build_tree_nodes(
            three_unit_kb, [f1, f2], annotations_f1_f2, max_schema_facets=2
        )
        facets_by_id = {"f1": f1, "f2": f2}
        tree = prune_and_connect(
            nodes, three_unit_kb, facets_by_id, annotations_f1_f2, h_memo,
            refinement_threshold=0.3,
            metric_mode="entropy",
        )
        surviving_ids = {n.id for n in tree.nodes}
        # f1 (smaller) is accepted first; with low tau, it captures f2 and f1|f2
        assert "" in surviving_ids
        assert "f1" in surviving_ids
        # Negative: confirm f2 and f1|f2 were pruned
        assert "f2" not in surviving_ids
        assert "f1|f2" not in surviving_ids
        # With high tau, f2 survives because f1 doesn't capture enough of it
        tree_high = prune_and_connect(
            nodes, three_unit_kb, facets_by_id, annotations_f1_f2, h_memo,
            refinement_threshold=0.9,
            metric_mode="entropy",
        )
        high_ids = {n.id for n in tree_high.nodes}
        assert "f2" in high_ids

    def test_evaluate_schemas_records_rejections(
        self, three_unit_kb, f1, f2, annotations_f1_f2
    ):
        """Schema evaluation exposes rejected schemas separately from the tree."""
        nodes, h_memo = build_tree_nodes(
            three_unit_kb, [f1, f2], annotations_f1_f2, max_schema_facets=2
        )
        evaluation = evaluate_schemas(
            nodes, three_unit_kb, {"f1": f1, "f2": f2},
            annotations_f1_f2, h_memo,
            refinement_threshold=0.3,
            metric_mode="entropy",
        )
        surviving_ids = {schema.id for schema in evaluation.accepted_schemas}
        rejected_by_id = {
            schema.id: schema.rejected_by_id
            for schema in evaluation.rejected_schemas
        }

        assert {node.id for node in evaluation.tree.nodes} == surviving_ids
        assert "f2" in rejected_by_id
        assert rejected_by_id["f2"] in surviving_ids
        assert rejected_by_id["f2"] == "f1"
        assert evaluation.rejected_schemas[0].refinement_threshold == 0.3

    def test_independent_nodes_survive(self, three_unit_kb, f1, f2, annotations_f1_f2):
        """
        With a very high tau (close to 1), almost nothing gets pruned.
        Independent nodes should both survive.
        """
        nodes, h_memo = build_tree_nodes(
            three_unit_kb, [f1, f2], annotations_f1_f2, max_schema_facets=2
        )
        facets_by_id = {"f1": f1, "f2": f2}
        tree = prune_and_connect(
            nodes, three_unit_kb, facets_by_id, annotations_f1_f2, h_memo,
            refinement_threshold=0.99,
            metric_mode="entropy",
        )
        surviving_ids = {n.id for n in tree.nodes}
        # With tau=0.99, most nodes survive since MI/H rarely exceeds 0.99
        assert "f1" in surviving_ids
        assert "f2" in surviving_ids

    def test_edge_direction(self, three_unit_kb, f1, f2, annotations_f1_f2):
        """Edges should go from smaller to larger size."""
        # Use high tau so multiple nodes survive and produce edges
        nodes, h_memo = build_tree_nodes(
            three_unit_kb, [f1, f2], annotations_f1_f2, max_schema_facets=2
        )
        facets_by_id = {"f1": f1, "f2": f2}
        tree = prune_and_connect(
            nodes, three_unit_kb, facets_by_id, annotations_f1_f2, h_memo,
            refinement_threshold=0.99,
            metric_mode="entropy",
        )
        assert len(tree.edges) > 0, "Need at least one edge to test direction"
        node_by_id = {n.id: n for n in tree.nodes}
        for edge in tree.edges:
            source = node_by_id[edge.source_id]
            target = node_by_id[edge.target_id]
            assert source.size < target.size, (
                f"Edge {edge.source_id}->{edge.target_id}: "
                f"source.size={source.size} should be < target.size={target.size}"
            )

    def test_edge_metadata(self, three_unit_kb, f1, f2, annotations_f1_f2):
        """Edges should have correct added_facets and marginal gains."""
        nodes, h_memo = build_tree_nodes(
            three_unit_kb, [f1, f2], annotations_f1_f2, max_schema_facets=2
        )
        facets_by_id = {"f1": f1, "f2": f2}
        tree = prune_and_connect(
            nodes, three_unit_kb, facets_by_id, annotations_f1_f2, h_memo,
            refinement_threshold=0.99,
            metric_mode="entropy",
        )
        assert len(tree.edges) > 0, "Need at least one edge to test metadata"
        for edge in tree.edges:
            source_fids = set(
                [n for n in tree.nodes if n.id == edge.source_id][0].facet_ids
            )
            target_fids = set(
                [n for n in tree.nodes if n.id == edge.target_id][0].facet_ids
            )
            assert set(edge.added_facets) == target_fids - source_fids
            assert set(edge.removed_facets) == source_fids - target_fids

    def test_empty_node_survives(self, three_unit_kb, f1, f2, annotations_f1_f2):
        """The empty schema node (no facets) should always survive."""
        nodes, h_memo = build_tree_nodes(
            three_unit_kb, [f1, f2], annotations_f1_f2, max_schema_facets=2
        )
        facets_by_id = {"f1": f1, "f2": f2}
        tree = prune_and_connect(
            nodes, three_unit_kb, facets_by_id, annotations_f1_f2, h_memo,
            refinement_threshold=0.5,
            metric_mode="entropy",
        )
        surviving_ids = {n.id for n in tree.nodes}
        assert "" in surviving_ids

    def test_same_size_keeps_better_quality(
        self, three_unit_kb, f1, f3, annotations_f1_f2_f3
    ):
        """
        f1 and f3 have size=2 and identical partitions (f3 duplicates f1).
        Bidirectionally redundant. Only one should survive.
        """
        nodes, h_memo = build_tree_nodes(
            three_unit_kb, [f1, f3], annotations_f1_f2_f3,
            max_schema_facets=2,
        )
        facets_by_id = {"f1": f1, "f3": f3}
        tree = prune_and_connect(
            nodes, three_unit_kb, facets_by_id,
            annotations_f1_f2_f3, h_memo,
            refinement_threshold=0.5,
            metric_mode="entropy",
        )
        surviving_ids = {n.id for n in tree.nodes}
        # f1 and f3 have identical partitions → bidirectionally redundant
        # Only one should survive (the first in sort order)
        assert not ("f1" in surviving_ids and "f3" in surviving_ids), (
            "Both duplicate facets survived — one should be pruned"
        )
        assert "f1" in surviving_ids or "f3" in surviving_ids

    def test_independent_survive_moderate_threshold(
        self, six_unit_kb, f_temporal, f_content, independent_annotations
    ):
        """
        f_temporal and f_content create orthogonal partitions.
        MI(f_temporal, f_content) = 0. Both should survive at any tau > 0.
        """
        nodes, h_memo = build_tree_nodes(
            six_unit_kb, [f_temporal, f_content], independent_annotations,
            max_schema_facets=2,
        )
        facets_by_id = {"f_temporal": f_temporal, "f_content": f_content}
        tree = prune_and_connect(
            nodes, six_unit_kb, facets_by_id,
            independent_annotations, h_memo,
            refinement_threshold=0.5,
            metric_mode="entropy",
        )
        surviving_ids = {n.id for n in tree.nodes}
        # Both should survive — they're independent (MI=0)
        assert "f_temporal" in surviving_ids
        assert "f_content" in surviving_ids

    def test_independent_mi_is_zero(
        self, six_unit_kb, f_temporal, f_content, independent_annotations
    ):
        """Verify that MI between truly orthogonal facets is 0."""
        cells_t = compute_cells(
            six_unit_kb, [f_temporal],
            independent_annotations.filter_facets(["f_temporal"]),
        )
        cells_c = compute_cells(
            six_unit_kb, [f_content],
            independent_annotations.filter_facets(["f_content"]),
        )
        cells_union = compute_cells(
            six_unit_kb, [f_temporal, f_content],
            independent_annotations.filter_facets(["f_temporal", "f_content"]),
        )
        h_t = compute_signature_entropy(cells_t)
        h_c = compute_signature_entropy(cells_c)
        h_union = compute_signature_entropy(cells_union)
        mi = compute_node_mi(h_t, h_c, h_union)
        assert math.isclose(mi, 0.0, abs_tol=1e-9), (
            f"MI should be 0 for independent facets, got {mi}"
        )

    def test_independent_joint_has_edges(
        self, six_unit_kb, f_temporal, f_content, independent_annotations
    ):
        """
        With independent facets at high tau, the joint schema survives and
        has edges from single-facet parents.

        At tau=0.5, f_content captures 61% of the joint (log2(3)/log2(6)),
        so the joint gets pruned. At tau=0.7, neither single facet captures
        enough and the joint survives.
        """
        nodes, h_memo = build_tree_nodes(
            six_unit_kb, [f_temporal, f_content], independent_annotations,
            max_schema_facets=2,
        )
        facets_by_id = {"f_temporal": f_temporal, "f_content": f_content}
        tree = prune_and_connect(
            nodes, six_unit_kb, facets_by_id,
            independent_annotations, h_memo,
            refinement_threshold=0.7,
            metric_mode="entropy",
        )
        surviving_ids = {n.id for n in tree.nodes}
        assert "f_content|f_temporal" in surviving_ids
        # Both single-facet nodes should have edges to the joint node
        edge_pairs = {(e.source_id, e.target_id) for e in tree.edges}
        assert ("f_temporal", "f_content|f_temporal") in edge_pairs or \
               ("f_content", "f_content|f_temporal") in edge_pairs

    def test_independent_joint_pruned_at_low_tau(
        self, six_unit_kb, f_temporal, f_content, independent_annotations
    ):
        """
        At tau=0.5, f_content captures 61% of the joint → joint is pruned.
        Demonstrates that tau controls how aggressively the tree prunes.
        """
        nodes, h_memo = build_tree_nodes(
            six_unit_kb, [f_temporal, f_content], independent_annotations,
            max_schema_facets=2,
        )
        facets_by_id = {"f_temporal": f_temporal, "f_content": f_content}
        tree = prune_and_connect(
            nodes, six_unit_kb, facets_by_id,
            independent_annotations, h_memo,
            refinement_threshold=0.5,
            metric_mode="entropy",
        )
        surviving_ids = {n.id for n in tree.nodes}
        # Joint is pruned at low tau
        assert "f_content|f_temporal" not in surviving_ids
        # But both individual facets survive (they're independent)
        assert "f_temporal" in surviving_ids
        assert "f_content" in surviving_ids


# ============= Tests: transitive reduction =============

class TestTransitiveReduction:
    def test_chain_removes_shortcut(self):
        """A->B->C chain: if A->C also exists, it should be removed."""
        nodes = [
            make_tree_node([], size=0, D=1.0, H=1.0),
            make_tree_node(["f1"], size=2, D=0.5, H=1.5),
            make_tree_node(["f1", "f2"], size=5, D=0.0, H=2.0),
        ]
        edges = [
            TreeEdge("", "f1", MI=0.5, refinement_score=0.5,
                     marginal_D_gain=0.5, marginal_eta2_gain=0.0,
                     added_facets=["f1"], removed_facets=[]),
            TreeEdge("f1", "f1|f2", MI=0.8, refinement_score=0.53,
                     marginal_D_gain=0.5, marginal_eta2_gain=0.0,
                     added_facets=["f2"], removed_facets=[]),
            TreeEdge("", "f1|f2", MI=0.5, refinement_score=0.5,
                     marginal_D_gain=1.0, marginal_eta2_gain=0.0,
                     added_facets=["f1", "f2"], removed_facets=[]),
        ]
        reduced = _transitive_reduction(nodes, edges)
        # The shortcut "" -> "f1|f2" should be removed since "" -> "f1" -> "f1|f2" exists
        targets = [(e.source_id, e.target_id) for e in reduced]
        assert ("", "f1") in targets
        assert ("f1", "f1|f2") in targets
        assert ("", "f1|f2") not in targets

    def test_no_reduction_needed(self):
        """If no shortcuts exist, all edges are kept."""
        nodes = [
            make_tree_node([], size=0),
            make_tree_node(["f1"], size=2),
            make_tree_node(["f2"], size=3),
        ]
        edges = [
            TreeEdge("", "f1", MI=0.5, refinement_score=0.5,
                     marginal_D_gain=0.5, marginal_eta2_gain=0.0,
                     added_facets=["f1"], removed_facets=[]),
            TreeEdge("", "f2", MI=0.5, refinement_score=0.5,
                     marginal_D_gain=0.3, marginal_eta2_gain=0.0,
                     added_facets=["f2"], removed_facets=[]),
        ]
        reduced = _transitive_reduction(nodes, edges)
        assert len(reduced) == 2

    def test_diamond_reduction(self):
        """
        Diamond: A->B, A->C, B->D, C->D.
        A->D shortcut should be removed (if it exists), but A->B and A->C stay.
        """
        nodes = [
            make_tree_node([], size=0),
            make_tree_node(["f1"], size=2),
            make_tree_node(["f2"], size=3),
            make_tree_node(["f1", "f2"], size=5),
        ]
        edges = [
            TreeEdge("", "f1", MI=0.5, refinement_score=0.5,
                     marginal_D_gain=0.3, marginal_eta2_gain=0.0,
                     added_facets=["f1"], removed_facets=[]),
            TreeEdge("", "f2", MI=0.4, refinement_score=0.4,
                     marginal_D_gain=0.2, marginal_eta2_gain=0.0,
                     added_facets=["f2"], removed_facets=[]),
            TreeEdge("f1", "f1|f2", MI=0.6, refinement_score=0.4,
                     marginal_D_gain=0.5, marginal_eta2_gain=0.0,
                     added_facets=["f2"], removed_facets=[]),
            TreeEdge("f2", "f1|f2", MI=0.6, refinement_score=0.5,
                     marginal_D_gain=0.3, marginal_eta2_gain=0.0,
                     added_facets=["f1"], removed_facets=[]),
            TreeEdge("", "f1|f2", MI=0.7, refinement_score=0.7,
                     marginal_D_gain=1.0, marginal_eta2_gain=0.0,
                     added_facets=["f1", "f2"], removed_facets=[]),
        ]
        reduced = _transitive_reduction(nodes, edges)
        targets = [(e.source_id, e.target_id) for e in reduced]
        # "" -> "f1|f2" is a shortcut (reachable via "" -> "f1" -> "f1|f2")
        assert ("", "f1|f2") not in targets
        assert ("", "f1") in targets
        assert ("", "f2") in targets
        assert ("f1", "f1|f2") in targets
        assert ("f2", "f1|f2") in targets


# ============= Tests: is_tree_improved =============

class TestCompareTreeQuality:
    def test_more_nodes_wins(self):
        """Tree with more surviving nodes is better."""
        a = RefinementTree(
            nodes=[
                make_tree_node(["f1"], size=2, D=0.5),
                make_tree_node(["f2"], size=3, D=0.3),
            ],
            edges=[],
        )
        b = RefinementTree(
            nodes=[make_tree_node(["f1"], size=2, D=0.5)],
            edges=[],
        )
        assert compare_tree_quality(a, b) == 1
        assert compare_tree_quality(b, a) == -1

    def test_same_nodes_fewer_facets_wins(self):
        """Same node count, fewer unique facets is better."""
        # a: 2 nodes using facets f1, f2 (2 unique)
        a = RefinementTree(
            nodes=[
                make_tree_node(["f1"], size=2, D=0.5),
                make_tree_node(["f2"], size=3, D=0.3),
            ],
            edges=[],
        )
        # b: 2 nodes using facets f1, f2, f3 (3 unique)
        b = RefinementTree(
            nodes=[
                make_tree_node(["f1"], size=2, D=0.5),
                make_tree_node(["f2", "f3"], size=4, D=0.2),
            ],
            edges=[],
        )
        assert compare_tree_quality(a, b) == 1
        assert compare_tree_quality(b, a) == -1

    def test_equal_trees(self):
        """Same node count and same unique facets → tie."""
        tree = RefinementTree(
            nodes=[make_tree_node(["f1"], size=2, D=0.5)],
            edges=[],
        )
        assert compare_tree_quality(tree, tree) == 0

    def test_empty_trees(self):
        """Two empty trees are equal."""
        a = RefinementTree(nodes=[], edges=[])
        b = RefinementTree(nodes=[], edges=[])
        assert compare_tree_quality(a, b) == 0


# ============= Tests: serialization =============

class TestSerialization:
    def test_tree_node_roundtrip(self):
        node = make_tree_node(["f1", "f2"], size=5, D=0.3, eta2=0.7, H=1.5)
        restored = TreeNode.from_dict(node.to_dict())
        assert restored.id == node.id
        assert restored.facet_ids == node.facet_ids
        assert restored.D == node.D
        assert restored.H == node.H

    def test_tree_edge_roundtrip(self):
        edge = TreeEdge(
            source_id="f1", target_id="f1|f2",
            MI=0.5, refinement_score=0.8,
            marginal_D_gain=0.3, marginal_eta2_gain=0.1,
            added_facets=["f2"], removed_facets=[],
            edge_type="add",
        )
        restored = TreeEdge.from_dict(edge.to_dict())
        assert restored.source_id == edge.source_id
        assert restored.MI == edge.MI
        assert restored.added_facets == edge.added_facets

    def test_refinement_tree_roundtrip(self):
        tree = RefinementTree(
            nodes=[make_tree_node(["f1"], size=2, D=0.5, H=1.0)],
            edges=[TreeEdge("", "f1", MI=0.5, refinement_score=0.5,
                           marginal_D_gain=0.5, marginal_eta2_gain=0.0,
                           added_facets=["f1"], removed_facets=[])],
        )
        restored = RefinementTree.from_dict(tree.to_dict())
        assert len(restored.nodes) == 1
        assert len(restored.edges) == 1
        assert restored.nodes[0].id == "f1"

    def test_schema_evaluation_roundtrip(self):
        tree = RefinementTree(
            nodes=[make_tree_node(["f1"], size=2, D=0.5, H=1.0)],
            edges=[],
        )
        evaluation = SchemaEvaluation(
            tree=tree,
            accepted_schemas=[],
            rejected_schemas=[],
        )
        restored = SchemaEvaluation.from_dict(evaluation.to_dict())
        assert restored.tree.nodes[0].id == "f1"
        assert restored.accepted_schemas == []
        assert restored.rejected_schemas == []

    def test_discriminativeness_property(self):
        """TreeNode.discriminativeness should return D."""
        node = make_tree_node(["f1"], D=0.42)
        assert node.discriminativeness == 0.42


# ============= Tests: integration =============

class TestIntegration:
    def test_full_pipeline(self, three_unit_kb, f1, f2, annotations_f1_f2):
        """Build nodes → prune → connect → sample → check improvement."""
        facets = [f1, f2]
        facets_by_id = {f.id: f for f in facets}

        # Build
        nodes, h_memo = build_tree_nodes(
            three_unit_kb, facets, annotations_f1_f2, max_schema_facets=2
        )
        assert len(nodes) == 4

        # Prune and connect
        tree = prune_and_connect(
            nodes, three_unit_kb, facets_by_id, annotations_f1_f2, h_memo,
            refinement_threshold=0.5, metric_mode="entropy",
        )
        assert len(tree.nodes) > 0

        # All surviving nodes should have valid metrics
        for node in tree.nodes:
            assert node.D >= 0
            assert node.H >= 0
            assert node.size >= 0

        # Edges should connect surviving nodes only
        surviving_ids = {n.id for n in tree.nodes}
        for edge in tree.edges:
            assert edge.source_id in surviving_ids
            assert edge.target_id in surviving_ids
