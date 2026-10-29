"""Pool selection: unconditional add below capacity, strict-improvement replace at capacity."""

from __future__ import annotations

from contextual.lifecycle import ProcessedCandidate
from contextual.evaluation import PoolEvaluator
from contextual.models import CandidateStatus, RunState

from tests.contextual.helpers import label_book, make_settings, make_facet, make_kb


def _processed(fid: str) -> ProcessedCandidate:
    return ProcessedCandidate(
        facet=make_facet(fid), annotations=label_book([fid]), embedding=[1.0],
    )


def _evaluator(pool_size: int) -> PoolEvaluator:
    return PoolEvaluator(config=make_settings(pool_size=pool_size), focused_kb=make_kb())


def _fill(evaluator: PoolEvaluator, *fids: str) -> RunState:
    state = RunState()
    for fid in fids:
        decision = evaluator.consider(state, _processed(fid))
        assert decision.status == CandidateStatus.ACCEPTED
    return state


def test_unconditional_add_below_capacity_even_if_redundant():
    ev = _evaluator(pool_size=3)
    state = RunState()

    d1 = ev.consider(state, _processed("A"))
    assert d1.status == CandidateStatus.ACCEPTED and d1.improved
    assert d1.snapshot.tree_nodes == 2

    # A2 is redundant with A: tree does not grow, but a non-full pool still takes it.
    d2 = ev.consider(state, _processed("A2"))
    assert d2.status == CandidateStatus.ACCEPTED and d2.improved
    assert d2.snapshot.tree_nodes == 2
    assert state.pool_ids() == ["A", "A2"]
    assert d2.snapshot.pool_ids == ["A", "A2"]


def test_replace_at_capacity_requires_strict_improvement_first_index_wins():
    ev = _evaluator(pool_size=2)
    state = _fill(ev, "A", "A2")  # 2 nodes

    d = ev.consider(state, _processed("B"))
    # Replacing A or A2 both yield 3 nodes; first index wins.
    assert d.status == CandidateStatus.REPLACED
    assert d.replaced_facet_id == "A"
    assert state.pool_ids() == ["B", "A2"]
    assert d.snapshot.tree_nodes == 3
    assert set(state.annotations.facet_ids()) == {"B", "A2"}


def test_tie_keeps_incumbent():
    ev = _evaluator(pool_size=2)
    state = _fill(ev, "A", "B")  # 3 nodes

    # A3 replacing A gives an equal tree; replacing B gives a worse one.
    d = ev.consider(state, _processed("A3"))
    assert d.status == CandidateStatus.SCHEMA_REJECTED
    assert not d.improved
    assert state.pool_ids() == ["A", "B"]
    assert d.snapshot.tree_nodes == 3


def test_reject_non_improving_candidate_at_capacity():
    ev = _evaluator(pool_size=2)
    state = _fill(ev, "A", "B")

    # C is independent of both, but at capacity it can only swap in, which ties.
    d = ev.consider(state, _processed("C"))
    assert d.status == CandidateStatus.SCHEMA_REJECTED
    assert state.pool_ids() == ["A", "B"]
    assert "C" not in state.annotations.facet_ids()


def test_accepted_and_rejected_schema_summaries():
    ev = _evaluator(pool_size=3)
    state = _fill(ev, "A", "A2", "B")

    accepted_ids = {tuple(s.facet_ids) for s in state.accepted_schemas}
    assert ("B",) in accepted_ids
    assert len(state.rejected_schemas) == 1
    rejected = state.rejected_schemas[0]
    assert set(rejected.facet_ids) | set(rejected.rejected_by_facet_ids) == {"A", "A2"}
    assert rejected.refinement_score > state_threshold(ev)


def state_threshold(ev: PoolEvaluator) -> float:
    return ev.config.refinement_threshold
