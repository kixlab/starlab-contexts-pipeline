"""
Unit tests for the optimization module.

Tests metrics, Pareto frontier, and collision detection with concrete,
hand-verifiable examples.

Run with: python -m pytest task_rep/tests/test_optimization.py -v
"""

from __future__ import annotations

import math
import pytest

from task_rep.knowledge_items.schema import (
    KnowledgeBase,
    KnowledgeItem,
    KnowledgeItemType,
    KnowledgeSource,
)
from task_rep.facets.schema import AnnotationTable, Facet, FacetLabel
from task_rep.optimization.metrics import (
    compute_cells,
    compute_compactness,
    compute_discriminativeness,
    count_unique_units,
)
from task_rep.optimization.frontier import (
    FrontierPoint,
    clip_frontier,
    compute_frontier,
    is_pareto_improved,
    sample_schemas,
)
from task_rep.optimization.collision import find_colliding_cells, sample_target_items


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


# ============= compute_discriminativeness =============

class TestComputeDiscriminativeness:
    def test_zero_units(self):
        assert compute_discriminativeness({}, num_units=0) == 0.0

    def test_empty_cells(self):
        assert compute_discriminativeness({}, num_units=3) == 0.0

    def test_perfect_separation(self):
        """Each unit in its own cell → D = 0."""
        cells = {"a": {"u1"}, "b": {"u2"}, "c": {"u3"}}
        assert compute_discriminativeness(cells, num_units=3) == pytest.approx(0.0)

    def test_no_separation(self):
        """All units in one cell → D = log2(N)."""
        cells = {"all": {"u1", "u2", "u3"}}
        expected = math.log(3, 2)
        assert compute_discriminativeness(cells, num_units=3) == pytest.approx(expected)

    def test_partial_separation(self):
        """
        2 units in cell A, 1 in cell B (N=3):
        D = (2/3)*log2(2) + (1/3)*log2(1) = 2/3.
        """
        cells = {"A": {"u1", "u2"}, "B": {"u3"}}
        expected = (2 / 3) * math.log(2, 2)  # = 2/3
        assert compute_discriminativeness(cells, num_units=3) == pytest.approx(expected)

    def test_custom_base(self):
        """D uses the supplied logarithm base."""
        cells = {"all": {"u1", "u2", "u3", "u4"}}
        expected_base2 = math.log(4, 2)   # = 2.0
        expected_base10 = math.log(4, 10)
        assert compute_discriminativeness(cells, num_units=4, base=2) == pytest.approx(expected_base2)
        assert compute_discriminativeness(cells, num_units=4, base=10) == pytest.approx(expected_base10)

    def test_d_increases_with_worse_separation(self):
        """More collision → higher D."""
        perfect = {"a": {"u1"}, "b": {"u2"}, "c": {"u3"}}
        partial = {"a": {"u1"}, "b": {"u2", "u3"}}
        none = {"a": {"u1", "u2", "u3"}}
        d_perfect = compute_discriminativeness(perfect, 3)
        d_partial = compute_discriminativeness(partial, 3)
        d_none = compute_discriminativeness(none, 3)
        assert d_perfect < d_partial < d_none


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
        assert cells[""] == {"u1", "u2", "u3"}

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
        assert singleton_cells[next(iter(singleton_cells))] == {"u1"}
        assert collision_cells[next(iter(collision_cells))] == {"u2", "u3"}

    def test_perfect_separation_with_f2(self, three_unit_kb, f2, annotations_f1_f2):
        """Under f2: every unit gets a distinct label → 3 singleton cells."""
        cells = compute_cells(three_unit_kb, [f2], annotations_f1_f2)
        assert len(cells) == 3
        for sig, units in cells.items():
            assert len(units) == 1

    def test_subtype_prefix_separates_different_subtypes(self):
        """Items with different subtypes never share a cell even with identical labels."""
        from task_rep.knowledge_items.taxonomy import AtomicType
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
        assert cells[next(iter(cells))] == {"u1"}


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

    def test_better_d_but_much_higher_c_no_improvement(self):
        """New extends compactness far beyond old — not counted as improvement."""
        old = [make_frontier_point(d=0.5, c=2)]
        new = [make_frontier_point(d=0.3, c=10)]
        assert is_pareto_improved(old, new) is False

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


# ============= find_colliding_cells =============

class TestFindCollidingCells:
    def test_no_collisions(self):
        cells = {"a": {"u1"}, "b": {"u2"}, "c": {"u3"}}
        assert find_colliding_cells(cells) == {}

    def test_all_collisions(self):
        cells = {"a": {"u1", "u2"}, "b": {"u3", "u4"}}
        result = find_colliding_cells(cells)
        assert set(result.keys()) == {"a", "b"}

    def test_mixed(self):
        cells = {"a": {"u1", "u2"}, "b": {"u3"}}
        result = find_colliding_cells(cells)
        assert "a" in result
        assert "b" not in result

    def test_empty(self):
        assert find_colliding_cells({}) == {}

    def test_large_collision(self):
        cells = {"big": {"u1", "u2", "u3", "u4", "u5"}}
        result = find_colliding_cells(cells)
        assert result["big"] == {"u1", "u2", "u3", "u4", "u5"}


# ============= sample_target_items =============

class TestSampleTargetItems:
    def test_no_collisions_returns_empty(self, three_unit_kb, f1, annotations_f1_f2):
        result = sample_target_items(three_unit_kb, [f1], annotations_f1_f2, {})
        assert result == []

    def test_returns_one_result_per_cell(self, three_unit_kb, f1, annotations_f1_f2):
        """
        Cell '<Difficulty:b>' has {u2,u3} — one colliding cell.
        Should return one (target_ids, cell_sig) tuple.
        """
        cells = compute_cells(three_unit_kb, [f1], annotations_f1_f2)
        colliding = find_colliding_cells(cells)
        result = sample_target_items(
            three_unit_kb, [f1], annotations_f1_f2, colliding, max_cells=1
        )
        assert len(result) == 1
        target_ids, cell_sig = result[0]
        assert len(target_ids) == 2  # one rep per unit (u2 and u3)
        assert set(target_ids) == {"item2", "item3"}

    def test_max_cells_limits_output(self, f1, annotations_f1_f2):
        """max_cells=0 means no cells are processed."""
        kb = KnowledgeBase(items=[
            make_item("i1", unit_id="u1"),
            make_item("i2", unit_id="u2"),
            make_item("i3", unit_id="u3"),
        ])
        ann = AnnotationTable()
        ann[("i1", "f1")] = "a"; ann[("i2", "f1")] = "a"; ann[("i3", "f1")] = "a"
        cells = compute_cells(kb, [f1], ann)
        colliding = find_colliding_cells(cells)
        result = sample_target_items(kb, [f1], ann, colliding, max_cells=0)
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
        cells = compute_cells(kb, [f1], ann)
        # compute_cells groups by unit_id so this won't actually collide,
        # but we can test directly with a synthetic collision of 1 item
        fake_colliding = {"fake_sig": {"u1", "u2"}}
        result = sample_target_items(kb, [f1], ann, fake_colliding, max_cells=1)
        # fake_sig doesn't exist in cell_items → 0 representatives → skipped
        assert result == []

    def test_largest_cell_sampled_first(self, f1):
        """Cells are sorted by collision size, largest first."""
        kb = KnowledgeBase(items=[
            make_item("i1", unit_id="u1"),
            make_item("i2", unit_id="u2"),
            make_item("i3", unit_id="u3"),
            make_item("i4", unit_id="u4"),
        ])
        ann = AnnotationTable()
        # Cell "a": {u1,u2,u3} — size 3
        ann[("i1", "f1")] = "a"; ann[("i2", "f1")] = "a"; ann[("i3", "f1")] = "a"
        # Cell "b": {u4} — not a collision on its own, but we construct synthetic
        ann[("i4", "f1")] = "b"

        cells = compute_cells(kb, [f1], ann)
        colliding = find_colliding_cells(cells)  # only "a" is colliding

        # Add a second fake collision of size 2 for ordering test
        colliding_with_extra = dict(colliding)
        colliding_with_extra["<Difficulty:b>"] = {"u4", "u5"}  # size 2

        result = sample_target_items(
            kb, [f1], ann, colliding_with_extra, max_cells=1
        )
        assert len(result) == 1
        # Should pick the larger cell (size 3) first
        _, sig = result[0]
        assert "a" in sig

    def test_max_items_per_cell_limits_representatives(self, f1):
        """If a cell has more units than max_items_per_cell, subsample."""
        items = [make_item(f"i{k}", unit_id=f"u{k}") for k in range(6)]
        kb = KnowledgeBase(items=items)
        ann = AnnotationTable()
        for item in items:
            ann[(item.id, "f1")] = "same"

        cells = compute_cells(kb, [f1], ann)
        colliding = find_colliding_cells(cells)

        result = sample_target_items(
            kb, [f1], ann, colliding, max_cells=1, max_items_per_cell=3
        )
        assert len(result) == 1
        target_ids, _ = result[0]
        assert len(target_ids) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
