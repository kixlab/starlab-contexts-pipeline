"""Runner: counters, stopping, checkpoints, a0 batch semantics, iterative context."""

from __future__ import annotations

import pytest

from pipeline import candidates as candidates_mod
from pipeline.config import A2_CONTEXT, ContextComponent, ProposalStrategy
from pipeline.candidates import ProcessedCandidate
from pipeline.models import CandidateStatus, PipelineState, StopReason
from pipeline.runner import PipelineRunner

from tests.pipeline.helpers import (
    ScriptedLabeler,
    ScriptedProposer,
    ScriptedVocabBuilder,
    TableEmbedder,
    embedding_of,
    label_book,
    make_config,
    make_facet,
    make_kb,
)


def _runner(tmp_path, proposer, *, embedder=None, vocab=None, labeler=None, **overrides):
    return PipelineRunner(
        config=make_config(**overrides),
        task="test task",
        knowledge_base=make_kb(),
        proposer=proposer,
        vocab_builder=vocab or ScriptedVocabBuilder(),
        labeler=labeler or ScriptedLabeler(),
        embedder=embedder or TableEmbedder(),
        cache_dir=tmp_path,
    )


def _statuses(result_or_state):
    return [r.status for r in result_or_state.ledger]


# ---------------------------------------------------------------------------
# Stopping conditions
# ---------------------------------------------------------------------------

def test_budget_exhaustion(tmp_path):
    proposer = ScriptedProposer([[make_facet(f)] for f in ["A", "B", "C", "D"]])
    result = _runner(tmp_path, proposer, proposal_budget=3, pool_size=3).run()

    assert result.stop_reason == StopReason.PROPOSAL_BUDGET
    assert result.proposals == 3
    assert _statuses(result) == [CandidateStatus.ACCEPTED] * 3
    assert len(proposer.iterative_contexts) == 3


def test_attempt_limit(tmp_path):
    result = _runner(tmp_path, ScriptedProposer([]), attempt_limit=4,
                     refusal_patience=100).run()
    assert result.stop_reason == StopReason.ATTEMPT_LIMIT
    assert result.attempts == 4


def test_attempt_limit_defaults_to_three_times_budget():
    assert make_config(proposal_budget=7, pool_size=7).attempt_limit == 21


def test_refusal_patience(tmp_path):
    result = _runner(tmp_path, ScriptedProposer([]), refusal_patience=3).run()
    assert result.stop_reason == StopReason.REFUSAL_PATIENCE
    assert result.attempts == 3
    assert _statuses(result) == [CandidateStatus.REFUSED] * 3


def test_refusal_streak_resets_on_invalid(tmp_path):
    script = [[], [], [make_facet("X", title="")], [], []]
    result = _runner(tmp_path, ScriptedProposer(script), refusal_patience=3,
                     attempt_limit=5).run()
    # streak: 1, 2, reset(0), 1, 2 -> never reaches 3; attempt limit stops it.
    assert result.stop_reason == StopReason.ATTEMPT_LIMIT


def test_no_improvement_patience_default_is_five(tmp_path):
    script = [[make_facet(f"F{n}")] for n in range(10)]
    vocab = ScriptedVocabBuilder(fail_ids={f"F{n}" for n in range(10)})
    result = _runner(tmp_path, ScriptedProposer(script), vocab=vocab).run()
    assert result.stop_reason == StopReason.NO_IMPROVEMENT_PATIENCE
    assert result.attempts == 5


def test_stop_precedence_budget_over_no_improvement(tmp_path, monkeypatch):
    monkeypatch.setattr(candidates_mod, "compute_irr",
                        lambda **kw: {"low_irr_facet_ids": ["P"]})
    script = [[make_facet("A")], [make_facet("B")], [make_facet("P")]]
    result = _runner(tmp_path, ScriptedProposer(script), proposal_budget=3,
                     pool_size=3, no_improvement_patience=1,
                     irr_sample_videos=3).run()
    # Step 3 trips both proposal budget (3) and no-improvement patience (1).
    assert result.stop_reason == StopReason.PROPOSAL_BUDGET


def test_stop_precedence_attempts_over_refusals(tmp_path):
    result = _runner(tmp_path, ScriptedProposer([]), attempt_limit=2,
                     refusal_patience=2).run()
    assert result.stop_reason == StopReason.ATTEMPT_LIMIT


# ---------------------------------------------------------------------------
# Counter table (normative): one attempt per outcome, exact counter deltas
# ---------------------------------------------------------------------------

# (attempts, proposals, refusal streak: +1/"reset", no-improve: +1/"reset"/0)
COUNTER_TABLE = {
    CandidateStatus.REFUSED: (1, 0, +1, 0),
    CandidateStatus.INVALID: (1, 0, "reset", 0),
    CandidateStatus.DUPLICATE: (1, 0, "reset", +1),
    CandidateStatus.IRR_PRUNED: (1, 1, "reset", +1),
    CandidateStatus.VOCABULARY_FAILED: (1, 1, "reset", +1),
    CandidateStatus.LABELING_FAILED: (1, 1, "reset", +1),
    CandidateStatus.SCHEMA_REJECTED: (1, 1, "reset", +1),
    CandidateStatus.ACCEPTED: (1, 1, "reset", "reset"),
    CandidateStatus.REPLACED: (1, 1, "reset", "reset"),
}


def _setup_outcome(tmp_path, monkeypatch, status):
    """Return (runner, state) such that one iteration yields ``status``."""
    emb = TableEmbedder({"A": [1, 0], "Dup": [1, 0]})
    vocab = ScriptedVocabBuilder()
    labeler = ScriptedLabeler(partition_of={"Dup": "aaaabbbb"})
    overrides = dict(proposal_budget=20, pool_size=20, refusal_patience=50,
                     no_improvement_patience=50, attempt_limit=60)
    facet = make_facet("C")
    state = PipelineState()

    if status == CandidateStatus.REFUSED:
        script = [[]]
    elif status == CandidateStatus.INVALID:
        script = [[make_facet("X", title="")]]
    elif status == CandidateStatus.DUPLICATE:
        script = [[make_facet("Dup")]]
        state.embeddings["A"] = embedding_of(emb, make_facet("A"))
    elif status == CandidateStatus.IRR_PRUNED:
        monkeypatch.setattr(candidates_mod, "compute_irr",
                            lambda **kw: {"low_irr_facet_ids": ["C"]})
        overrides["irr_sample_videos"] = 3
        script = [[facet]]
    elif status == CandidateStatus.VOCABULARY_FAILED:
        vocab = ScriptedVocabBuilder(fail_ids={"C"})
        script = [[facet]]
    elif status == CandidateStatus.LABELING_FAILED:
        labeler = ScriptedLabeler(fail_ids={"C"})
        script = [[facet]]
    elif status in (CandidateStatus.SCHEMA_REJECTED, CandidateStatus.REPLACED):
        overrides["pool_size"] = 2
        script = [[make_facet("D")]] if status == CandidateStatus.SCHEMA_REJECTED \
            else [[make_facet("B")]]
    else:  # ACCEPTED
        script = [[facet]]

    runner = _runner(tmp_path, ScriptedProposer(script), embedder=emb,
                     vocab=vocab, labeler=labeler, **overrides)

    if status == CandidateStatus.SCHEMA_REJECTED:
        _seed_pool(runner, state, ["A", "B"])  # D (redundant with B) only ties
    elif status == CandidateStatus.REPLACED:
        _seed_pool(runner, state, ["A", "A2"])  # B strictly improves

    state.attempts, state.proposals = 10, 4
    state.refusal_streak, state.no_improve_streak = 2, 2
    return runner, state


def _seed_pool(runner, state, fids):
    for fid in fids:
        runner.evaluator.consider(state, ProcessedCandidate(
            facet=make_facet(fid), annotations=label_book([fid]), embedding=[0.0],
        ))


@pytest.mark.parametrize("status", list(COUNTER_TABLE))
def test_counter_table(tmp_path, monkeypatch, status):
    runner, state = _setup_outcome(tmp_path, monkeypatch, status)
    d_attempts, d_proposals, refusal, no_improve = COUNTER_TABLE[status]

    runner.run_iteration(state)

    assert state.ledger[-1].status == status
    assert state.attempts == 10 + d_attempts
    assert state.proposals == 4 + d_proposals
    assert state.refusal_streak == (3 if refusal == +1 else 0)
    expected_ni = {+1: 3, "reset": 0, 0: 2}[no_improve]
    assert state.no_improve_streak == expected_ni
    assert state.ledger[-1].counts_against_budget == bool(d_proposals)


def test_ledger_records_tree_quality_after_selection(tmp_path):
    script = [[make_facet("A")], [make_facet("B")]]
    result = _runner(tmp_path, ScriptedProposer(script), proposal_budget=2,
                     pool_size=2).run()
    assert [r.tree_quality.tree_nodes for r in result.ledger] == [2, 3]
    assert result.ledger[1].tree_quality.pool_ids == ["A", "B"]


def test_ledger_records_scope_definition_and_nearest_neighbor(tmp_path):
    emb = TableEmbedder({"A": [1, 0], "B": [4, 3]})
    script = [[make_facet("A")], [make_facet("B", definition="bee")]]
    result = _runner(tmp_path, ScriptedProposer(script), embedder=emb,
                     proposal_budget=2, pool_size=2).run()
    rec = result.ledger[1]
    assert rec.scope == "global" and rec.definition == "bee"
    assert rec.nearest_id == "A"
    assert rec.nearest_similarity == pytest.approx(0.8)


def test_duplicates_do_not_join_the_dedup_set(tmp_path):
    emb = TableEmbedder({"A": [1, 0], "A2": [1, 0]})
    script = [[make_facet("A")], [make_facet("A2")]]
    runner = _runner(tmp_path, ScriptedProposer(script), embedder=emb)
    state = PipelineState()
    runner.run_iteration(state)
    runner.run_iteration(state)
    assert _statuses(state) == [CandidateStatus.ACCEPTED, CandidateStatus.DUPLICATE]
    assert set(state.embeddings) == {"A"}


# ---------------------------------------------------------------------------
# Checkpoints (iterative)
# ---------------------------------------------------------------------------

def test_checkpoint_after_every_iteration(tmp_path):
    script = [[make_facet("A")], [], [make_facet("B")]]
    runner = _runner(tmp_path, ScriptedProposer(script), attempt_limit=3)
    saved = []
    original = runner.store.save_checkpoint

    def spy(state):
        saved.append(len(state.ledger))
        original(state)

    runner.store.save_checkpoint = spy
    runner.run()
    assert saved[:3] == [1, 2, 3]


def test_resume_after_crash_continues_ledger(tmp_path):
    first = ScriptedProposer([[make_facet("A")], [make_facet("B")],
                              RuntimeError("network down")])
    with pytest.raises(RuntimeError):
        _runner(tmp_path, first, proposal_budget=3, pool_size=3, use_cache=True).run()

    second = ScriptedProposer([[make_facet("C")]])
    result = _runner(tmp_path, second, proposal_budget=3, pool_size=3,
                     use_cache=True).run()

    assert [r.facet_id for r in result.ledger] == ["A", "B", "C"]
    assert result.stop_reason == StopReason.PROPOSAL_BUDGET
    assert len(second.iterative_contexts) == 1
    assert [f.id for f in second.iterative_contexts[0].pool] == ["A", "B"]


def test_completed_run_is_restored_without_proposing(tmp_path):
    script = [[make_facet("A")], [make_facet("B")]]
    _runner(tmp_path, ScriptedProposer(script), proposal_budget=2, pool_size=2,
            use_cache=True).run()

    again = ScriptedProposer([[make_facet("Z")]])
    result = _runner(tmp_path, again, proposal_budget=2, pool_size=2,
                     use_cache=True).run()
    assert again.iterative_contexts == []
    assert [r.facet_id for r in result.ledger] == ["A", "B"]


# ---------------------------------------------------------------------------
# Iterative context
# ---------------------------------------------------------------------------

def test_a1_context_has_items_and_pool(tmp_path):
    proposer = ScriptedProposer([[make_facet("A")], [make_facet("B")]])
    _runner(tmp_path, proposer, proposal_budget=2, pool_size=2).run()

    first, second = proposer.iterative_contexts
    assert first.components == frozenset({ContextComponent.FACETS, ContextComponent.ITEMS})
    assert first.pool == [] and first.videos  # empty-pool cold start, segments shown
    assert [f.id for f in second.pool] == ["A"]
    assert first.history == [] and second.history == []


def test_a2_context_has_no_segments(tmp_path):
    proposer = ScriptedProposer([[make_facet("A")], [make_facet("B")]])
    _runner(tmp_path, proposer, context=A2_CONTEXT, proposal_budget=2,
            pool_size=2).run()

    first, second = proposer.iterative_contexts
    assert first.components == frozenset({ContextComponent.SCHEMAS})
    assert first.videos is None
    assert first.pool == [] and first.accepted_schemas == []
    assert second.accepted_schemas  # schema feedback after the first facet


def test_history_lists_non_accepted_candidates_when_enabled(tmp_path):
    context = frozenset({ContextComponent.FACETS, ContextComponent.HISTORY})
    script = [[make_facet("A")], [make_facet("A2")], [make_facet("V")],
              [make_facet("B")], [make_facet("Z")]]
    proposer = ScriptedProposer(script)
    _runner(tmp_path, proposer, vocab=ScriptedVocabBuilder(fail_ids={"V"}),
            context=context, proposal_budget=5, pool_size=2).run()

    # Before the 5th proposal: A, A2 accepted; V vocabulary-failed; B replaced A.
    history = proposer.iterative_contexts[4].history
    outcomes = {h.title: h.outcome for h in history}
    assert outcomes == {"V": "vocabulary_failed", "A": "replaced_out"}
    entry = next(h for h in history if h.title == "V")
    assert entry.scope == "global" and entry.definition == "Definition of V"


def test_history_off_by_default(tmp_path):
    proposer = ScriptedProposer([[make_facet("V")], [make_facet("A")]])
    _runner(tmp_path, proposer, vocab=ScriptedVocabBuilder(fail_ids={"V"})).run()
    assert proposer.iterative_contexts[1].history == []


# ---------------------------------------------------------------------------
# a0 batch
# ---------------------------------------------------------------------------

def _batch_runner(tmp_path, batch, **overrides):
    proposer = ScriptedProposer(batch=batch)
    overrides.setdefault("strategy", ProposalStrategy.BATCH)
    return proposer, _runner(tmp_path, proposer, **overrides)


def test_a0_requests_budget_in_one_call(tmp_path):
    proposer, runner = _batch_runner(tmp_path, [make_facet("A"), make_facet("B")],
                                     proposal_budget=5, pool_size=5)
    result = runner.run()
    assert len(proposer.batch_contexts) == 1
    assert proposer.batch_contexts[0].num_facets == 5
    assert result.attempts == 1
    assert result.stop_reason == StopReason.BATCH_COMPLETE  # shortfall accepted


def test_a0_truncates_before_dedup(tmp_path):
    emb = TableEmbedder({"A": [1, 0], "A2": [1, 0]})
    batch = [make_facet(f) for f in ["A", "A2", "B", "C"]]
    _, runner = _batch_runner(tmp_path, batch, embedder=emb,
                              proposal_budget=3, pool_size=3)
    result = runner.run()

    assert _statuses(result) == [
        CandidateStatus.ACCEPTED, CandidateStatus.DUPLICATE,
        CandidateStatus.ACCEPTED, CandidateStatus.TRUNCATED,
    ]
    assert result.ledger[3].facet_id == "C"
    assert result.proposals == 2  # duplicate cost no budget; C was cut
    assert result.attempts == 1
    assert result.stop_reason == StopReason.BATCH_COMPLETE


def test_a0_truncated_rows_touch_no_counters(tmp_path):
    batch = [make_facet(f) for f in ["A", "B", "C"]]
    _, runner = _batch_runner(tmp_path, batch, proposal_budget=1, pool_size=1)
    result = runner.run()
    truncated = [r for r in result.ledger if r.status == CandidateStatus.TRUNCATED]
    assert len(truncated) == 2
    assert all(not r.counts_against_budget for r in truncated)
    assert result.attempts == 1 and result.proposals == 1


def test_a0_empty_response_is_refused(tmp_path):
    _, runner = _batch_runner(tmp_path, [])
    result = runner.run()
    assert _statuses(result) == [CandidateStatus.REFUSED]
    assert result.stop_reason == StopReason.BATCH_REFUSED
    assert result.attempts == 1


def test_a0_seeds_pool_then_greedily_replaces(tmp_path):
    batch = [make_facet(f) for f in ["A", "A2", "B"]]
    _, runner = _batch_runner(tmp_path, batch, proposal_budget=3, pool_size=2)
    result = runner.run()
    assert _statuses(result) == [CandidateStatus.ACCEPTED, CandidateStatus.ACCEPTED,
                                 CandidateStatus.REPLACED]
    assert [f.id for f in result.pool] == ["B", "A2"]


def test_a0_lifecycle_is_batched(tmp_path):
    vocab, labeler = ScriptedVocabBuilder(), ScriptedLabeler()
    batch = [make_facet(f) for f in ["A", "B", "C"]]
    _, runner = _batch_runner(tmp_path, batch, vocab=vocab, labeler=labeler,
                              proposal_budget=3, pool_size=3)
    runner.run()
    assert vocab.calls == [["A", "B", "C"]]
    assert labeler.calls == [["A", "B", "C"]]


def test_a0_is_atomic(tmp_path):
    batch = [make_facet(f) for f in ["A", "B", "C"]]
    _, runner = _batch_runner(tmp_path, batch, proposal_budget=3, pool_size=3)
    saved = []
    original = runner.store.save_checkpoint
    runner.store.save_checkpoint = lambda s: (saved.append(s.stop_reason), original(s))
    runner.run()
    assert saved == [StopReason.BATCH_COMPLETE]
