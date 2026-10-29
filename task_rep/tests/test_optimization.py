"""
Unit tests for the optimization module.

Tests schema metrics with concrete, hand-verifiable examples.

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
