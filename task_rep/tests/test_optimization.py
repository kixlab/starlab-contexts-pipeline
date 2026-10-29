"""
Unit tests for the optimization module.

Tests metrics, Pareto frontier, and collision detection with concrete,
hand-verifiable examples.

Run with: python -m pytest tests/test_optimization.py -v
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
from optimization.metrics import (
    compute_cells,
    compute_compactness,
    compute_discriminativeness,
    count_unique_units,
)
from optimization.frontier import (
    FrontierPoint,
    clip_frontier,
    compute_frontier,
    is_pareto_improved,
    sample_schemas,
)
from optimization.collision import sample_target_items


# ============= Helpers =============

def make_source(video_id: str = "v1") -> KnowledgeSource:
    return KnowledgeSource(video_id=video_id)


def make_item(
    id: str,
    unit_id: str | None = None,
    video_id: str = "v1",
    subtype=None,
) -> KnowledgeItem:
    return KnowledgeItem(
        id=id,
        type=KnowledgeItemType.ATOMIC,
        text=f"Item {id}",
        source=make_source(video_id),
        unit_id=unit_id,
        subtype=subtype,
    )


def make_facet(id: str, title: str, labels: list[str], scope: str = "global") -> Facet:
    return Facet(
        id=id,
        title=title,
        title_plural=title + "s",
        scope=scope,
        definition=f"Definition of {title}",
        justification="Testing",
        guidelines=[],
        vocabulary=[FacetLabel(label=lbl, definition=f"Def {lbl}") for lbl in labels],
    )


def make_frontier_point(d: float, c: int, facet_ids: list[str] | None = None) -> FrontierPoint:
    return FrontierPoint(discriminativeness=d, compactness=c, facet_ids=facet_ids or [])


# ============= Shared fixtures =============

@pytest.fixture
def three_unit_kb():
    """
    Knowledge base with 3 units (u1, u2, u3), one item each.

    item1 → unit u1
    item2 → unit u2
    item3 → unit u3
    """
    return KnowledgeBase(items=[
        make_item("item1", unit_id="u1"),
        make_item("item2", unit_id="u2"),
        make_item("item3", unit_id="u3"),
    ])

@pytest.fixture
def three_unit_kb_with_clash():
    """
    Knowledge base with 3 units (u1, u2, u3), one item each.
    item1 → unit u1
    item2 → unit u2
    item3 → unit u3
    item4 → unit u3
    """
    return KnowledgeBase(items=[
        make_item("item1", unit_id="u1"),
        make_item("item2", unit_id="u2"),
        make_item("item3", unit_id="u3"),
        make_item("item4", unit_id="u3"),
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
def annotations_f1_f2():
    """
    Annotation table for three_unit_kb with f1 and f2:

      item1: Difficulty=a, Tool=x  (unit u1 — unique on both)
      item2: Difficulty=b, Tool=y  (unit u2)
      item3: Difficulty=b, Tool=z  (unit u3 — same as u2 on Difficulty)

    Cells under {f1}:  "<Difficulty:a>"->{u1},  "<Difficulty:b>"->{u2,u3}
    Cells under {f2}:  "<Tool:x>"->{u1},  "<Tool:y>"->{u2},  "<Tool:z>"->{u3}
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
def annotations_f1_f2_with_clash():
    """
    Annotation table for three_unit_kb_with_clash with f1 and f2:
    """
    ann = AnnotationTable()
    ann[("item1", "f1")] = "a"
    ann[("item2", "f1")] = "b"
    ann[("item3", "f1")] = "b"
    ann[("item4", "f1")] = "a"
    ann[("item1", "f2")] = "x"
    ann[("item2", "f2")] = "y"
    ann[("item3", "f2")] = "z"
    ann[("item4", "f2")] = "z"
    return ann


# ============= compute_discriminativeness =============

class TestComputeDiscriminativeness:
    def test_zero_items(self):
        assert compute_discriminativeness({}) == 0.0

    def test_empty_cells(self):
        assert compute_discriminativeness({}) == 0.0

    def test_perfect_separation(self):
        """Each unit in its own cell → H(U|C) = 0."""
        cells = {"a": {"u1": 1}, "b": {"u2": 1}, "c": {"u3": 1}}
        assert compute_discriminativeness(cells) == pytest.approx(0.0)

    def test_no_separation(self):
        """All units in one cell (uniform) → H(U|C) = log2(N)."""
        cells = {"all": {"u1": 1, "u2": 1, "u3": 1}}
        expected = math.log(3, 2)
        assert compute_discriminativeness(cells) == pytest.approx(expected)

    def test_partial_separation(self):
        """
        2 units in cell A (1 item each), 1 in cell B (1 item):
        H(U|C) = (2/3)*log2(2) + (1/3)*0 = 2/3.
        """
        cells = {"A": {"u1": 1, "u2": 1}, "B": {"u3": 1}}
        expected = (2 / 3) * math.log(2, 2)  # = 2/3
        assert compute_discriminativeness(cells) == pytest.approx(expected)

    def test_custom_base(self):
        """H uses the supplied logarithm base."""
        cells = {"all": {"u1": 1, "u2": 1, "u3": 1, "u4": 1}}
        expected_base2 = math.log(4, 2)   # = 2.0
        expected_base10 = math.log(4, 10)
        assert compute_discriminativeness(cells, base=2) == pytest.approx(expected_base2)
        assert compute_discriminativeness(cells, base=10) == pytest.approx(expected_base10)

    def test_d_increases_with_worse_separation(self):
        """More collision → higher H(U|C)."""
        perfect = {"a": {"u1": 1}, "b": {"u2": 1}, "c": {"u3": 1}}
        partial = {"a": {"u1": 1}, "b": {"u2": 1, "u3": 1}}
        none = {"a": {"u1": 1, "u2": 1, "u3": 1}}
        d_perfect = compute_discriminativeness(perfect)
        d_partial = compute_discriminativeness(partial)
        d_none = compute_discriminativeness(none)
        assert d_perfect < d_partial < d_none

    def test_unequal_item_counts_within_cell(self):
        """Under uniform-over-units, item counts per unit don't affect entropy."""
        # 100 items from u1, 1 from u2 → each unit still gets equal weight
        skewed = {"a": {"u1": 100, "u2": 1}}
        # 1 item each
        uniform = {"a": {"u1": 1, "u2": 1}}
        d_skewed = compute_discriminativeness(skewed)
        d_uniform = compute_discriminativeness(uniform)
        assert d_skewed == pytest.approx(d_uniform)

    def test_unit_overlap_across_cells(self):
        """
        Unit u3 spans two cells (items from different tutorials).
        Under uniform-over-units, u3's weight is split across cells.
        """
        # u3 has 1 item in each of two cells
        cells = {"a": {"u1": 1, "u3": 1}, "b": {"u2": 1, "u3": 1}}
        d = compute_discriminativeness(cells)
        # N=3, unit_totals: u1=1, u2=1, u3=2
        # Cell a: w(u1)=1, w(u3)=0.5, w_sum=1.5, P(u1|a)=2/3, P(u3|a)=1/3
        # Cell b: w(u2)=1, w(u3)=0.5, w_sum=1.5, P(u2|b)=2/3, P(u3|b)=1/3
        # H(U|C) = 2 * (1.5/3) * H(2/3,1/3) = log2(3) - 2/3
        expected = math.log(3, 2) - 2 / 3
        assert d == pytest.approx(expected)
        # H(U) under uniform prior = log2(N) = log2(3)
        h_u = math.log(3, 2)
        assert d <= h_u  # H(U|C) ≤ H(U), always


# ============= compute_compactness =============

class TestComputeCompactness:
    def test_empty(self):
        assert compute_compactness([]) == 0

    def test_single_facet(self, f1):
        """f1 has labels [a, b] → C = 2."""
        assert compute_compactness([f1]) == 2

    def test_two_facets(self, f1, f2):
        """f1=2 + f2=3 → C = 5."""
        assert compute_compactness([f1, f2]) == 5

    def test_na_label_excluded(self):
        """Labels named 'na' (any case) are excluded from compactness."""
        facet = make_facet("fx", "Phase", ["setup", "NA", "Na"])
        assert compute_compactness([facet]) == 1  # only "setup" counts

    def test_no_vocabulary(self):
        facet = make_facet("fx", "Phase", [])
        assert compute_compactness([facet]) == 0


# ============= count_unique_units =============

class TestCountUniqueUnits:
    def test_empty_kb(self):
        kb = KnowledgeBase(items=[])
        assert count_unique_units(kb) == 0

    def test_no_unit_ids(self):
        kb = KnowledgeBase(items=[make_item("i1"), make_item("i2")])
        assert count_unique_units(kb) == 0

    def test_counts_distinct_units(self, three_unit_kb):
        assert count_unique_units(three_unit_kb) == 3

    def test_multiple_items_per_unit(self):
        kb = KnowledgeBase(items=[
            make_item("i1", unit_id="u1"),
            make_item("i2", unit_id="u1"),
            make_item("i3", unit_id="u2"),
        ])
        assert count_unique_units(kb) == 2

    def test_mixed_none_and_units(self):
        kb = KnowledgeBase(items=[
            make_item("i1", unit_id="u1"),
            make_item("i2", unit_id=None),
            make_item("i3", unit_id="u2"),
        ])
        assert count_unique_units(kb) == 2


# ============= compute_cells =============

class TestComputeCells:
    def test_skips_items_without_unit_id(self, f1, annotations_f1_f2):
        kb = KnowledgeBase(items=[make_item("orphan", unit_id=None)])
        cells = compute_cells(kb, [f1], annotations_f1_f2)
        assert cells == {}

    def test_empty_facets_all_in_one_cell(self, three_unit_kb, annotations_f1_f2):
        """No facets → all items share the empty signature → one cell."""
        cells = compute_cells(three_unit_kb, [], annotations_f1_f2)
        assert len(cells) == 1
        assert cells[""] == {"u1": 1, "u2": 1, "u3": 1}

    def test_partial_separation_with_f1(self, three_unit_kb, f1, annotations_f1_f2):
        """
        Under f1: item1 gets 'a', items2+3 get 'b'.
        → 2 cells: one containing u1, one containing {u2,u3}.
        """
        cells = compute_cells(three_unit_kb, [f1], annotations_f1_f2)
        assert len(cells) == 2
        # Find the cells by their unit contents
        singleton_cells = {sig: units for sig, units in cells.items() if len(units) == 1}
        collision_cells = {sig: units for sig, units in cells.items() if len(units) > 1}
        assert len(singleton_cells) == 1
        assert len(collision_cells) == 1
        assert singleton_cells[next(iter(singleton_cells))] == {"u1": 1}
        assert collision_cells[next(iter(collision_cells))] == {"u2": 1, "u3": 1}

    def test_perfect_separation_with_f2(self, three_unit_kb, f2, annotations_f1_f2):
        """Under f2: every unit gets a distinct label → 3 singleton cells."""
        cells = compute_cells(three_unit_kb, [f2], annotations_f1_f2)
        assert len(cells) == 3
        for sig, units in cells.items():
            assert len(units) == 1

    def test_subtype_prefix_separates_different_subtypes(self):
        """Items with different subtypes never share a cell even with identical labels."""
        from knowledge_items.taxonomy import AtomicType
        item_a = make_item("ia", unit_id="u1", subtype=AtomicType.METHOD_SUBGOAL)
        item_b = make_item("ib", unit_id="u2", subtype="step_type")
        kb = KnowledgeBase(items=[item_a, item_b])

        facet = make_facet("fx", "Phase", ["setup"])
        ann = AnnotationTable()
        ann[("ia", "fx")] = "setup"
        ann[("ib", "fx")] = "setup"  # same label, different subtype

        cells = compute_cells(kb, [facet], ann)
        # Despite same facet label, subtypes differ → separate cells
        assert len(cells) == 2
        assert all(len(units) == 1 for units in cells.values())

    def test_multiple_items_same_unit_in_same_cell(self, f1):
        """Two items with the same unit_id and same label end up in the same cell."""
        kb = KnowledgeBase(items=[
            make_item("i1", unit_id="u1"),
            make_item("i2", unit_id="u1"),
        ])
        ann = AnnotationTable()
        ann[("i1", "f1")] = "a"
        ann[("i2", "f1")] = "a"
        cells = compute_cells(kb, [f1], ann)
        assert len(cells) == 1
        assert cells[next(iter(cells))] == {"u1": 2}


# ============= clip_frontier =============

class TestClipFrontier:
    def test_empty(self):
        assert clip_frontier([]) == []

    def test_removes_resolved_points(self):
        """Points with D ≤ min_d are removed."""
        frontier = [
            make_frontier_point(d=0.8, c=0),
            make_frontier_point(d=0.3, c=2),
            make_frontier_point(d=0.005, c=4),  # ≤ 0.01 → removed
        ]
        clipped = clip_frontier(frontier, min_d=0.01)
        assert len(clipped) == 2
        assert all(p.discriminativeness > 0.01 for p in clipped)

    def test_removes_near_duplicate_d(self):
        """Points with nearly-equal D to the previous kept point are skipped."""
        frontier = [
            make_frontier_point(d=0.8, c=0),
            make_frontier_point(d=0.8 + 1e-12, c=2),  # essentially same D
            make_frontier_point(d=0.4, c=4),
        ]
        clipped = clip_frontier(frontier)
        assert len(clipped) == 2
        assert clipped[0].compactness == 0
        assert clipped[1].compactness == 4

    def test_keeps_all_distinct_non_resolved(self):
        frontier = [
            make_frontier_point(d=1.0, c=0),
            make_frontier_point(d=0.5, c=2),
            make_frontier_point(d=0.1, c=4),
        ]
        clipped = clip_frontier(frontier, min_d=0.01)
        assert len(clipped) == 3

    def test_all_resolved(self):
        frontier = [
            make_frontier_point(d=0.001, c=0),
            make_frontier_point(d=0.0, c=2),
        ]
        assert clip_frontier(frontier) == []


# ============= is_pareto_improved =============

class TestIsParetoImproved:
    def test_both_empty_after_clip(self):
        """Fully-resolved old frontier → no meaningful improvement possible."""
        old = [make_frontier_point(d=0.0, c=2)]
        new = [make_frontier_point(d=0.5, c=1)]
        assert is_pareto_improved(old, new) is False

    def test_new_empty_after_clip(self):
        old = [make_frontier_point(d=0.5, c=2)]
        new = [make_frontier_point(d=0.0, c=2)]
        assert is_pareto_improved(old, new) is False

    def test_strictly_lower_d_at_same_c(self):
        old = [make_frontier_point(d=0.5, c=2)]
        new = [make_frontier_point(d=0.3, c=2)]
        assert is_pareto_improved(old, new) is True

    def test_same_d_at_strictly_lower_c(self):
        old = [make_frontier_point(d=0.5, c=3)]
        new = [make_frontier_point(d=0.5, c=2)]
        assert is_pareto_improved(old, new) is True

    def test_same_d_same_c_no_improvement(self):
        old = [make_frontier_point(d=0.5, c=2)]
        new = [make_frontier_point(d=0.5, c=2)]
        assert is_pareto_improved(old, new) is False

    def test_higher_d_at_same_c_no_improvement(self):
        old = [make_frontier_point(d=0.3, c=2)]
        new = [make_frontier_point(d=0.5, c=2)]
        assert is_pareto_improved(old, new) is False

    def test_extension_clipped_by_min_d_no_improvement(self):
        """Extension whose D falls below min_d is clipped out — not counted as improvement."""
        old = [make_frontier_point(d=0.5, c=2)]
        new = [make_frontier_point(d=0.3, c=10)]
        assert is_pareto_improved(old, new, min_d=0.35) is False

    def test_improvement_at_one_point_among_many(self):
        """Improvement at any paired point is sufficient."""
        old = [
            make_frontier_point(d=0.8, c=1),
            make_frontier_point(d=0.5, c=3),
        ]
        new = [
            make_frontier_point(d=0.8, c=1),
            make_frontier_point(d=0.3, c=3),  # better D at same C
        ]
        assert is_pareto_improved(old, new) is True

    def test_new_extends_frontier_with_lower_d(self):
        """New adds a higher-C point with lower D than old's minimum — real improvement."""
        old = [make_frontier_point(d=4.611, c=0)]
        new = [
            make_frontier_point(d=4.611, c=0),
            make_frontier_point(d=3.698, c=4),  # extends frontier, lower D
        ]
        assert is_pareto_improved(old, new) is True

    def test_new_extends_frontier_same_d_no_improvement(self):
        """New adds a higher-C point but D is no better than old's minimum — not an improvement."""
        old = [make_frontier_point(d=0.5, c=2), make_frontier_point(d=0.3, c=4)]
        new = [
            make_frontier_point(d=0.5, c=2),
            make_frontier_point(d=0.3, c=4),
            make_frontier_point(d=0.3, c=8),  # extends, but D = old minimum
        ]
        assert is_pareto_improved(old, new) is False


# ============= compute_frontier =============

class TestComputeFrontier:
    def test_no_facets_returns_single_point(self, three_unit_kb):
        """With no facets, the only schema is the empty one."""
        frontier = compute_frontier(three_unit_kb, [], AnnotationTable())
        assert len(frontier) == 1
        assert frontier[0].compactness == 0
        assert frontier[0].facet_ids == []

    def test_frontier_sorted_by_ascending_compactness(self, three_unit_kb, f1, f2, annotations_f1_f2):
        frontier = compute_frontier(three_unit_kb, [f1, f2], annotations_f1_f2)
        compactnesses = [p.compactness for p in frontier]
        assert compactnesses == sorted(compactnesses)

    def test_discriminativeness_strictly_decreasing(self, three_unit_kb, f1, f2, annotations_f1_f2):
        """D must strictly decrease as C increases (dominated points removed)."""
        frontier = compute_frontier(three_unit_kb, [f1, f2], annotations_f1_f2)
        d_values = [p.discriminativeness for p in frontier]
        for i in range(1, len(d_values)):
            assert d_values[i] < d_values[i - 1]

    def test_known_frontier_values(self, three_unit_kb, f1, f2, annotations_f1_f2):
        """
        Concrete verification:
          {} (C=0):     D = log2(3)  ≈ 1.585   (all 3 units in 1 cell)
          {f1} (C=2):   D = 2/3     ≈ 0.667   (cell 'b' has {u2,u3})
          {f2} (C=3):   D = 0                  (perfect separation)
          {f1,f2} (C=5): D = 0 → dominated by {f2} → not on frontier
        """
        frontier = compute_frontier(three_unit_kb, [f1, f2], annotations_f1_f2)

        by_c = {p.compactness: p for p in frontier}
        assert set(by_c.keys()) == {0, 2, 3}

        assert by_c[0].discriminativeness == pytest.approx(math.log(3, 2))
        assert by_c[2].discriminativeness == pytest.approx(2 / 3)
        assert by_c[3].discriminativeness == pytest.approx(0.0)

    def test_dominated_schema_excluded(self, three_unit_kb, f1, f2, annotations_f1_f2):
        """
        {f1,f2} at C=5 with D=0 is dominated by {f2} at C=3 with D=0.
        It must not appear on the frontier.
        """
        frontier = compute_frontier(three_unit_kb, [f1, f2], annotations_f1_f2)
        compactnesses = {p.compactness for p in frontier}
        assert 5 not in compactnesses

    def test_single_facet_frontier(self, three_unit_kb, f1, annotations_f1_f2):
        """
        One facet: two possible schemas ({} and {f1}).
        Both should appear since {f1} improves D.
        """
        ann = annotations_f1_f2.filter_facets(["f1"])
        frontier = compute_frontier(three_unit_kb, [f1], ann)
        assert len(frontier) == 2
        compactnesses = [p.compactness for p in frontier]
        assert 0 in compactnesses
        assert 2 in compactnesses

    def test_known_frontier_values_with_clash(self, three_unit_kb_with_clash, f1, f2, annotations_f1_f2_with_clash):
        """
        Concrete verification (3 unique units: u1, u2, u3; 4 items):

          item1→u1: f1=a, f2=x
          item2→u2: f1=b, f2=y
          item3→u3: f1=b, f2=z
          item4→u3: f1=a, f2=z  ← clash: u3 lands in two f1 cells

          Under uniform-over-units prior:

          {} (C=0):      H(U|C) = log2(3)
                          one cell: {u1:1, u2:1, u3:2}
                          w(u1)=1, w(u2)=1, w(u3)=1 → all equal → H = log2(3)
          {f1} (C=2):    H(U|C) = log2(3) - 2/3
                          cell <a> → {u1:1, u3:1}, cell <b> → {u2:1, u3:1}
                          u3 split across cells → same as overlap test
          {f2} (C=3):    H(U|C) = 0
                          cell <x>→{u1:1}, <y>→{u2:1}, <z>→{u3:2}  (each cell pure)
          {f1,f2} (C=5): H = 0 → dominated by {f2} → excluded from frontier
        """
        frontier = compute_frontier(three_unit_kb_with_clash, [f1, f2], annotations_f1_f2_with_clash)

        by_c = {p.compactness: p for p in frontier}
        assert set(by_c.keys()) == {0, 2, 3}

        assert by_c[0].discriminativeness == pytest.approx(math.log(3, 2))
        assert by_c[2].discriminativeness == pytest.approx(math.log(3, 2) - 2 / 3)
        assert by_c[3].discriminativeness == pytest.approx(0.0)

    def test_synergistic_facets_knapsack_regression(self):
        """
        Regression: the old 0/1 knapsack DP could miss the optimal subset
        when facets interact non-additively.

        Setup — 6 units (one item each), three 2-label facets (weight=2 each):

          Facet A: u1-u4 → a1  (4 units), u5-u6 → a2  (2 units)
            D({A}) = (4/6)·log2(4) + (2/6)·log2(2) = 4/3 + 1/3 = 5/3 ≈ 1.667

          Facet B: u1-u3 → b1  (3 units), u4-u6 → b2  (3 units)
            D({B}) = 2·(3/6)·log2(3) = log2(3) ≈ 1.585

          Facet C: u1-u2 → c1  (2 units), u3-u6 → c2  (4 units)
            D({C}) ≈ 1.667

        D({B}) < D({A}): the knapsack stores {B} at weight 2 and evicts {A},
        then tries {B,C} at weight 4 and never explores {A,C}.

          D({A,C}) = 3·(2/6)·log2(2) = 1.0       ← optimal at weight 4
          D({B,C}) = 1/3 + (1/2)·log2(3) ≈ 1.125 ← what the knapsack found

        Brute-force enumeration correctly identifies {A,C} as the winning
        schema at compactness 4.
        """
        kb = KnowledgeBase(items=[
            make_item("i1", unit_id="u1"),
            make_item("i2", unit_id="u2"),
            make_item("i3", unit_id="u3"),
            make_item("i4", unit_id="u4"),
            make_item("i5", unit_id="u5"),
            make_item("i6", unit_id="u6"),
        ])

        fa = make_facet("fa", "A", ["a1", "a2"])  # weight=2
        fb = make_facet("fb", "B", ["b1", "b2"])  # weight=2
        fc = make_facet("fc", "C", ["c1", "c2"])  # weight=2

        ann = AnnotationTable()
        for item_id in ("i1", "i2", "i3", "i4"):
            ann[(item_id, "fa")] = "a1"
        for item_id in ("i5", "i6"):
            ann[(item_id, "fa")] = "a2"

        for item_id in ("i1", "i2", "i3"):
            ann[(item_id, "fb")] = "b1"
        for item_id in ("i4", "i5", "i6"):
            ann[(item_id, "fb")] = "b2"

        for item_id in ("i1", "i2"):
            ann[(item_id, "fc")] = "c1"
        for item_id in ("i3", "i4", "i5", "i6"):
            ann[(item_id, "fc")] = "c2"

        frontier = compute_frontier(kb, [fa, fb, fc], ann)
        by_c = {p.compactness: p for p in frontier}

        # At compactness 4, the winner must be {A,C} (D=1.0), not {B,C} (D≈1.125).
        d_at_4 = by_c[4].discriminativeness
        d_ac = 1.0
        d_bc = 1 / 3 + math.log(3, 2) / 2
        assert d_at_4 == pytest.approx(d_ac), (
            f"Expected D({'{A,C}'})={d_ac:.4f} at compactness 4, got {d_at_4:.4f}. "
            f"Knapsack would have returned D({'{B,C}'})={d_bc:.4f}."
        )
        assert set(by_c[4].facet_ids) == {"fa", "fc"}


# ============= sample_schemas =============

class TestSampleSchemas:
    def test_empty_frontier(self, f1):
        result = sample_schemas([], {"f1": f1}, k=2)
        assert result == []

    def test_returns_k_schemas(self, f1, f2):
        frontier = [
            make_frontier_point(d=0.8, c=0, facet_ids=[]),
            make_frontier_point(d=0.5, c=2, facet_ids=["f1"]),
            make_frontier_point(d=0.2, c=5, facet_ids=["f1", "f2"]),
        ]
        result = sample_schemas(frontier, {"f1": f1, "f2": f2}, k=2)
        assert len(result) == 2

    def test_k_larger_than_frontier_returns_all(self, f1):
        frontier = [
            make_frontier_point(d=0.8, c=0, facet_ids=[]),
            make_frontier_point(d=0.4, c=2, facet_ids=["f1"]),
        ]
        result = sample_schemas(frontier, {"f1": f1}, k=10)
        assert len(result) == 2

    def test_resolves_facet_ids_to_objects(self, f1, f2):
        frontier = [make_frontier_point(d=0.5, c=5, facet_ids=["f1", "f2"])]
        result = sample_schemas(frontier, {"f1": f1, "f2": f2}, k=1)
        assert len(result) == 1
        facet_list, point = result[0]
        assert set(f.id for f in facet_list) == {"f1", "f2"}

    def test_missing_facet_id_skipped(self, f1):
        """Unknown facet_ids (not in facets_by_id) are silently skipped."""
        frontier = [make_frontier_point(d=0.5, c=5, facet_ids=["f1", "unknown"])]
        result = sample_schemas(frontier, {"f1": f1}, k=1)
        facet_list, _ = result[0]
        assert len(facet_list) == 1
        assert facet_list[0].id == "f1"


# ============= sample_target_items =============

class TestSampleTargetItems:
    def test_no_collisions_returns_empty(self, three_unit_kb, f1, annotations_f1_f2):
        """Single-unit cell → no collision → empty result."""
        # Only use f2 annotations which perfectly separate all 3 units
        f2 = make_facet("f2", "Tool", ["x", "y", "z"])
        result = sample_target_items(three_unit_kb, [f1, f2], annotations_f1_f2)
        assert result == []

    def test_returns_one_result_per_cell(self, three_unit_kb, f1, annotations_f1_f2):
        """
        Cell '<Difficulty:b>' has {u2,u3} — one colliding cell.
        Should return one (target_ids, cell_sig) tuple.
        """
        result = sample_target_items(
            three_unit_kb, [f1], annotations_f1_f2, max_cells=1
        )
        assert len(result) == 1
        target_ids, cell_sig = result[0]
        assert len(target_ids) == 2  # one rep per unit (u2 and u3)
        assert set(target_ids) == {"item2", "item3"}

    def test_max_cells_limits_output(self, f1):
        """max_cells=0 means no cells are processed."""
        kb = KnowledgeBase(items=[
            make_item("i1", unit_id="u1"),
            make_item("i2", unit_id="u2"),
            make_item("i3", unit_id="u3"),
        ])
        ann = AnnotationTable()
        ann[("i1", "f1")] = "a"; ann[("i2", "f1")] = "a"; ann[("i3", "f1")] = "a"
        result = sample_target_items(kb, [f1], ann, max_cells=0)
        assert result == []

    def test_skips_cells_with_fewer_than_two_representatives(self, f1):
        """A collision cell where only one unit has items is skipped (nothing to discriminate)."""
        # Two items from the same unit, same label → cell has 1 unit (not a real collision)
        kb = KnowledgeBase(items=[
            make_item("i1", unit_id="u1"),
            make_item("i2", unit_id="u1"),
        ])
        ann = AnnotationTable()
        ann[("i1", "f1")] = "a"
        ann[("i2", "f1")] = "a"
        result = sample_target_items(kb, [f1], ann, max_cells=1)
        assert result == []

    def test_largest_cell_sampled_first(self):
        """Cells are sorted by collision size, largest first."""
        # Two facets: f_coarse puts {u1,u2,u3} together, f_fine splits u3+u4 into their own cell
        f_coarse = make_facet("fc", "Coarse", ["a", "b"])
        f_fine = make_facet("ff", "Fine", ["x", "y"])
        kb = KnowledgeBase(items=[
            make_item("i1", unit_id="u1"),
            make_item("i2", unit_id="u2"),
            make_item("i3", unit_id="u3"),
            make_item("i4", unit_id="u4"),
            make_item("i5", unit_id="u5"),
        ])
        ann = AnnotationTable()
        # Under {fc}: cell "a" has {u1,u2,u3} (size 3), cell "b" has {u4,u5} (size 2)
        ann[("i1", "fc")] = "a"; ann[("i2", "fc")] = "a"; ann[("i3", "fc")] = "a"
        ann[("i4", "fc")] = "b"; ann[("i5", "fc")] = "b"

        result = sample_target_items(kb, [f_coarse], ann, max_cells=1, max_items_per_cell=5)
        assert len(result) == 1
        # Should pick the larger cell (size 3) first
        target_ids, _ = result[0]
        assert len(target_ids) == 3

    def test_max_items_per_cell_limits_representatives(self, f1):
        """If a cell has more units than max_items_per_cell, subsample."""
        items = [make_item(f"i{k}", unit_id=f"u{k}") for k in range(6)]
        kb = KnowledgeBase(items=items)
        ann = AnnotationTable()
        for item in items:
            ann[(item.id, "f1")] = "same"

        result = sample_target_items(
            kb, [f1], ann, max_cells=1, max_items_per_cell=3
        )
        assert len(result) == 1
        target_ids, _ = result[0]
        assert len(target_ids) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
