"""Runner: counters, stopping, checkpoints, a0 batch semantics, iterative context."""

from __future__ import annotations

import pytest

from facets import Facet

from contextual import lifecycle as candidates_mod
from contextual.settings import stage_seed
from pipeline.config import A2_CONTEXT, ContextComponent, ProposalStrategy
from contextual.lifecycle import ProcessedCandidate
from contextual.models import CandidateStatus, RunState, StopReason
from pipeline.runner import PipelineRunner
from utils import llm as llm_mod

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


def _runner(tmp_path, proposer, *, embedder=None, vocab=None, labeler=None, kb=None,
            **overrides):
    return PipelineRunner(
        config=make_config(**overrides),
        task="test task",
        knowledge_base=kb or make_kb(),
        proposer=proposer,
        vocab_builder=vocab or ScriptedVocabBuilder(),
        labeler=labeler or ScriptedLabeler(),
        embedder=embedder or TableEmbedder(),
        runs_dir=tmp_path,
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
    CandidateStatus.DUPLICATE: (1, 1, "reset", +1),   # a parseable proposal: uses budget
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
    state = RunState()

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
    state = RunState()
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
    runner.run()
    steps = runner.folder.steps()       # one appended line per attempt
    assert [len(s["ledger"]) for s in steps] == [1, 1, 1]
    assert [s["counters"]["attempts"] for s in steps] == [1, 2, 3]
    assert (runner.folder.path / "result.json").exists()


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


class CtrlC(ScriptedProposer):
    """Ctrl+C while the second proposal is being processed."""

    def propose_one(self, context):
        if len(self.iterative_contexts) == 1:
            raise KeyboardInterrupt
        return super().propose_one(context)


def test_interrupted_run_resumes_and_redoes_the_attempt_in_flight(tmp_path):
    first = _runner(tmp_path, CtrlC([[make_facet("A")]]),
                    proposal_budget=2, pool_size=2, use_cache=True)
    with pytest.raises(KeyboardInterrupt):
        first.run()
    assert "Interrupted during attempt 2" in first.folder.log_path.read_text()

    second = _runner(tmp_path, ScriptedProposer([[make_facet("B")]]), proposal_budget=2,
                     pool_size=2, use_cache=True)
    result = second.run()
    assert second.folder.path == first.folder.path and second.folder.resumed
    assert [r.facet_id for r in result.ledger] == ["A", "B"]
    assert "resumed after 1 step(s)" in second.folder.log_path.read_text()


def test_finished_run_is_not_reused_by_a_new_run(tmp_path):
    first = _runner(tmp_path, ScriptedProposer([[make_facet("A")], [make_facet("B")]]),
                    proposal_budget=2, pool_size=2, use_cache=True)
    first.run()
    again = ScriptedProposer([[make_facet("Y")], [make_facet("Z")]])
    second = _runner(tmp_path, again, proposal_budget=2, pool_size=2, use_cache=True)
    result = second.run()
    assert second.folder.path != first.folder.path and not second.folder.resumed
    assert [r.facet_id for r in result.ledger] == ["Y", "Z"]


# ---------------------------------------------------------------------------
# Iterative context
# ---------------------------------------------------------------------------

def test_a1_context_has_items_and_pool(tmp_path):
    proposer = ScriptedProposer([[make_facet("A")], [make_facet("B")]])
    _runner(tmp_path, proposer, proposal_budget=2, pool_size=2).run()

    first, second = proposer.iterative_contexts
    assert first.components == frozenset({ContextComponent.FACETS, ContextComponent.ITEMS, ContextComponent.HISTORY})
    assert first.pool == [] and sorted(first.corpus) == ["v1", "v2"]
    assert first.item_video_ids is None  # ITEMS shows every video
    assert [f.id for f in second.pool] == ["A"] and second.ids == {"A": "F1"}
    assert first.history == [] and second.history == []  # nothing failed yet


def test_a2_context_has_schemas_and_pool(tmp_path):
    proposer = ScriptedProposer([[make_facet("A")], [make_facet("B")]])
    _runner(tmp_path, proposer, context=A2_CONTEXT, proposal_budget=2,
            pool_size=2).run()

    first, second = proposer.iterative_contexts
    assert first.components == A2_CONTEXT
    assert first.pool == [] and first.accepted_schemas == []
    assert sorted(first.corpus) == ["v1", "v2"]  # every condition gets the tutorials
    assert second.accepted_schemas  # schema feedback after the first facet
    assert second.refinement_threshold == 0.2


def test_factor_ids_are_stable_across_replacement(tmp_path):
    # A, A2 enter (F1, F2); V fails; B replaces A in its pool slot and becomes F3.
    script = [[make_facet("A")], [make_facet("A2")], [make_facet("V")],
              [make_facet("B")], [make_facet("Z")]]
    proposer = ScriptedProposer(script)
    _runner(tmp_path, proposer, vocab=ScriptedVocabBuilder(fail_ids={"V"}),
            proposal_budget=5, pool_size=2).run()
    fifth = proposer.iterative_contexts[4]
    assert [f.id for f in fifth.pool] == ["B", "A2"]
    assert fifth.ids == {"A": "F1", "A2": "F2", "B": "F3"}


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
    assert outcomes == {"V": "no usable values could be built from the tutorials",
                        "A": 'was F1; replaced in the pool by F3 ("B")'}
    entry = next(h for h in history if h.title == "V")
    assert entry.scope == "global" and entry.definition == "Definition of V"


def test_history_off_without_component(tmp_path):
    proposer = ScriptedProposer([[make_facet("V")], [make_facet("A")]])
    _runner(tmp_path, proposer, vocab=ScriptedVocabBuilder(fail_ids={"V"}),
            context=frozenset({ContextComponent.FACETS, ContextComponent.ITEMS})).run()
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
    assert result.proposals == 3  # A, the duplicate, and B all use budget; C was cut
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
    runner.run()
    steps = runner.folder.steps()          # the whole batch is one step
    assert [s["stop_reason"] for s in steps] == ["batch_complete"]
    assert len(steps[0]["ledger"]) == 3


# ---------------------------------------------------------------------------
# Sampling seeds
# ---------------------------------------------------------------------------

class SeedEchoProposer:
    """Deterministic like a seeded model: the proposal depends only on the seed."""

    def __init__(self):
        self.seeds = []

    def propose_batch(self, context):
        self.seeds.append(context.seed)
        return [make_facet(f"S{context.seed}")]

    def propose_one(self, context):
        self.seeds.append(context.seed)
        return [make_facet(f"S{context.seed}")]


def test_failure_does_not_lock_iterative_run_into_repeats(tmp_path):
    """Regression: a fixed seed + unchanged prompt re-proposed the same facet
    after any failure, so the run looped on duplicates until patience ran out."""
    proposer = SeedEchoProposer()
    first = stage_seed(0, "proposal", 1)
    vocab = ScriptedVocabBuilder(fail_ids={f"S{first}"})
    result = _runner(tmp_path, proposer, vocab=vocab, proposal_budget=3,
                     pool_size=3).run()
    assert _statuses(result) == [CandidateStatus.VOCABULARY_FAILED,
                                 CandidateStatus.ACCEPTED, CandidateStatus.ACCEPTED]
    assert len(set(proposer.seeds)) == 3


def test_seeds_are_reproducible_and_differ_by_run_seed(tmp_path):
    runs = []
    for run_seed, sub in [(0, "a"), (0, "b"), (1, "c")]:
        proposer = SeedEchoProposer()
        _runner(tmp_path / sub, proposer, proposal_budget=2, pool_size=2,
                seed=run_seed).run()
        runs.append(proposer.seeds)
    assert runs[0] == runs[1]                  # same run seed: same sequence
    assert not set(runs[0]) & set(runs[2])     # different run seed: disjoint
    assert runs[0] == [stage_seed(0, "proposal", 1), stage_seed(0, "proposal", 2)]


def test_a0_proposal_uses_run_seed(tmp_path):
    proposer = SeedEchoProposer()
    _runner(tmp_path, proposer, strategy=ProposalStrategy.BATCH, seed=2).run()
    assert proposer.seeds == [stage_seed(2, "proposal")]


# ---------------------------------------------------------------------------
# Failures, outages, token usage
# ---------------------------------------------------------------------------


class OutageLabeler(ScriptedLabeler):
    """Raises an outage on the ``fail_on``-th labeling call (1-based)."""

    def __init__(self, fail_on):
        super().__init__()
        self.fail_on = fail_on

    def label_batch(self, task, facets, knowledge_base, temperature=None, seed=None):
        if len(self.calls) + 1 == self.fail_on:
            self.calls.append(None)
            raise llm_mod.LLMBatchOutage("all requests failed")
        return super().label_batch(task, facets, knowledge_base)


def test_outage_stops_run_without_recording_and_resumes(tmp_path):
    script = [[make_facet("A")], [make_facet("B")]]
    with pytest.raises(llm_mod.LLMBatchOutage):
        _runner(tmp_path, ScriptedProposer(script), labeler=OutageLabeler(fail_on=2),
                proposal_budget=3, pool_size=3, use_cache=True).run()

    resumed = _runner(tmp_path, ScriptedProposer([[make_facet("B")], [make_facet("C")]]),
                      proposal_budget=3, pool_size=3, use_cache=True).run()
    # The outage attempt left no trace; B is proposed again and accepted.
    assert [r.facet_id for r in resumed.ledger] == ["A", "B", "C"]
    assert all(r.status == CandidateStatus.ACCEPTED for r in resumed.ledger)


class UsageReportingLabeler(ScriptedLabeler):
    """Reports usage the way batch_run_llm_calls does (current usage stage)."""

    def label_batch(self, task, facets, knowledge_base, temperature=None, seed=None):
        llm_mod.USAGE.add(llm_mod._usage_stage.get(),
                          {"input_tokens": 10, "output_tokens": 2, "thinking_tokens": 1})
        return super().label_batch(task, facets, knowledge_base)


class UsageReportingProposer(ScriptedProposer):
    def propose_one(self, context):
        llm_mod.USAGE.add(llm_mod._usage_stage.get(),
                          {"input_tokens": 1000, "output_tokens": 50, "thinking_tokens": 20})
        return super().propose_one(context)


def test_usage_is_accumulated_per_stage_and_checkpointed(tmp_path):
    runner = _runner(tmp_path, UsageReportingProposer([[make_facet("A")], [], [make_facet("B")]]),
                     labeler=UsageReportingLabeler(), proposal_budget=2, pool_size=2,
                     use_cache=True)
    result = runner.run()
    assert result.usage == {
        "proposal": {"input_tokens": 3000, "output_tokens": 150,
                     "thinking_tokens": 60, "requests": 3},
        "labeling": {"input_tokens": 20, "output_tokens": 4,
                     "thinking_tokens": 2, "requests": 2},
    }
    assert runner.folder.steps()[-1]["usage"] == result.usage


class GappyVocab(ScriptedVocabBuilder):
    def build_vocabulary(self, task, facets, knowledge_base, sample_videos=0, seed=None):
        out = super().build_vocabulary(task, facets, knowledge_base, sample_videos)
        return [Facet.from_dict({**f.to_dict(), "metadata": {"vocab_proposal_failures": 2}})
                for f in out]


def test_vocabulary_gaps_are_noted_in_ledger(tmp_path):
    result = _runner(tmp_path, ScriptedProposer([[make_facet("A")]]), vocab=GappyVocab(),
                     proposal_budget=1, pool_size=1).run()
    assert result.ledger[0].status == CandidateStatus.ACCEPTED
    assert result.ledger[0].note == "vocabulary proposals failed for 2 video(s)"


# ---------------------------------------------------------------------------
# ITEMS sampling
# ---------------------------------------------------------------------------

def test_items_sampling_draws_per_attempt_and_leaves_other_components(tmp_path):
    proposer = ScriptedProposer([[make_facet("A")], [make_facet("B")]])
    _runner(tmp_path, proposer, items_sample_videos=1, proposal_budget=2, pool_size=2).run()
    for ctx in proposer.iterative_contexts:
        assert len(ctx.item_video_ids) == 1         # labels of 1 of the 2 videos
        assert len(ctx.corpus) == 2                 # the tutorials are never sampled
        assert len(ctx.knowledge_base.by_video()) == 2  # frequencies use the whole KB


def test_items_sampling_is_reproducible_and_off_by_default(tmp_path):
    picks = []
    for sub in ("a", "b"):
        proposer = ScriptedProposer([[make_facet("A")], [make_facet("B")]])
        _runner(tmp_path / sub, proposer, items_sample_videos=1, proposal_budget=2,
                pool_size=2).run()
        picks.append([c.item_video_ids for c in proposer.iterative_contexts])
    assert picks[0] == picks[1]

    proposer = ScriptedProposer([[make_facet("A")]])
    _runner(tmp_path / "c", proposer, proposal_budget=1, pool_size=1).run()
    assert proposer.iterative_contexts[0].item_video_ids is None


def test_corpus_sampling_restricts_what_the_runner_sees(tmp_path):
    proposer = ScriptedProposer([[make_facet("A")]])
    runner = _runner(tmp_path, proposer, corpus_sample_videos=1, proposal_budget=1,
                     pool_size=1)
    result = runner.run()
    assert len(runner.knowledge_base.by_video()) == 1
    assert len(proposer.iterative_contexts[0].corpus) == 1
    assert result.num_units == 4


def test_segments_outside_the_evaluated_set_are_never_shown(tmp_path):
    kb = make_kb()
    kb.items[0].unit_id = None  # i1 belongs to no knowledge unit
    proposer = ScriptedProposer([[make_facet("A")]])
    runner = _runner(tmp_path, proposer, kb=kb, proposal_budget=1, pool_size=1)
    runner.run()
    shown = [i.id for items in proposer.iterative_contexts[0].corpus.values() for i in items]
    assert "i1" not in shown and len(shown) == 7
    assert "i1" in [i.id for i in runner.knowledge_base]  # still labeled, for context


class LoggedProposer(ScriptedProposer):
    """Runs one (empty) logged batch per proposal, like ``VLMFacetProposer``."""

    def propose_one(self, context):
        llm_mod.batch_run_llm_calls(f"facet_iterative_{len(self.iterative_contexts) + 1}",
                                    [], lambda **_: {}, None, max_concurrency=1,
                                    model_config=None)
        return super().propose_one(context)


def test_each_step_records_its_request_logs(tmp_path):
    runner = _runner(tmp_path, LoggedProposer([[make_facet("A")], [], [make_facet("B")]]),
                     proposal_budget=2, pool_size=2)
    result = runner.run()
    assert result.runner_state["batches"] == [
        {"proposal": ["facet_iterative_1"]}, {"proposal": ["facet_iterative_2"]},
        {"proposal": ["facet_iterative_3"]}]
    assert "lm-batches proposal: facet_iterative_2" in runner.folder.log_path.read_text()
