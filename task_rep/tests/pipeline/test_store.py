"""Versioned checkpoints: exact-fingerprint restore and serialization round-trip."""

from __future__ import annotations

from knowledge_items.schema import KnowledgeBase

from pipeline.config import A2_CONTEXT
from pipeline.models import (
    CandidateRecord,
    CandidateStatus,
    PipelineResult,
    PipelineState,
    StopReason,
    TreeQualitySnapshot,
)
from pipeline.candidates import ProcessedCandidate
from pipeline.evaluation import PoolEvaluator
from pipeline.store import PipelineStore, kb_fingerprint

from tests.pipeline.helpers import label_book, make_config, make_facet, make_kb


def _state() -> PipelineState:
    cfg = make_config(pool_size=3)
    state = PipelineState()
    ev = PoolEvaluator(config=cfg, focused_kb=make_kb())
    for fid in ["A", "A2", "B"]:
        ev.consider(state, ProcessedCandidate(
            facet=make_facet(fid), annotations=label_book([fid]), embedding=[1.0, 0.0],
        ))
    state.embeddings = {"A": [1.0, 0.0], "A2": [0.6, 0.8], "B": [0.0, 1.0]}
    state.ledger = [
        CandidateRecord(
            attempt_index=1, status=CandidateStatus.ACCEPTED, facet_id="A", title="A",
            scope="global", definition="Definition of A", counts_against_budget=True,
            tree_quality=TreeQualitySnapshot(2, 1, ["A"]),
        ),
        CandidateRecord(
            attempt_index=2, status=CandidateStatus.DUPLICATE, facet_id="Ax", title="Ax",
            scope="global", definition="d", duplicate_of="A", nearest_id="A",
            nearest_similarity=0.97,
        ),
        CandidateRecord(
            attempt_index=3, status=CandidateStatus.TRUNCATED, facet_id="T", title="T",
        ),
    ]
    state.attempts, state.proposals, state.iteration = 3, 1, 3
    state.refusal_streak, state.no_improve_streak = 0, 1
    state.stop_reason = StopReason.PROPOSAL_BUDGET
    return state


def _store(tmp_path, config=None, kb_fp="kb1", prompt_fp="p1") -> PipelineStore:
    return PipelineStore(cache_dir=tmp_path, config=config or make_config(),
                         kb_fp=kb_fp, prompt_fp=prompt_fp)


def test_exact_fingerprint_restores(tmp_path):
    state = _state()
    _store(tmp_path).save_checkpoint(state)
    restored = _store(tmp_path).restore_checkpoint()
    assert restored is not None
    assert restored.to_dict() == state.to_dict()


def test_serialization_round_trip_keeps_ledger_and_embeddings(tmp_path):
    state = _state()
    _store(tmp_path).save_checkpoint(state)
    restored = _store(tmp_path).restore_checkpoint()

    assert restored.embeddings == state.embeddings
    assert restored.ledger[0].tree_quality.pool_ids == ["A"]
    assert restored.ledger[0].definition == "Definition of A"
    assert restored.ledger[1].nearest_similarity == 0.97
    assert restored.ledger[1].duplicate_of == "A"
    assert restored.ledger[2].status == CandidateStatus.TRUNCATED
    assert len(restored.rejected_schemas) == 1
    assert restored.pool_ids() == ["A", "A2", "B"]


def test_result_round_trip(tmp_path):
    result = PipelineResult.from_state(_state(), kb_fingerprint="kb1")
    path = _store(tmp_path).save_result(result)
    assert PipelineStore.load_result(path).to_dict() == result.to_dict()


def test_config_mismatch_invalidates(tmp_path):
    _store(tmp_path).save_checkpoint(_state())
    other = _store(tmp_path, config=make_config(context=A2_CONTEXT))
    assert other.restore_checkpoint() is None


def test_prompt_mismatch_invalidates(tmp_path):
    _store(tmp_path).save_checkpoint(_state())
    assert _store(tmp_path, prompt_fp="p2").restore_checkpoint() is None


def test_kb_mismatch_invalidates(tmp_path):
    _store(tmp_path).save_checkpoint(_state())
    assert _store(tmp_path, kb_fp="kb2").restore_checkpoint() is None


def test_tampered_checkpoint_fingerprint_is_ignored(tmp_path):
    store = _store(tmp_path)
    store.save_checkpoint(_state())
    path = store.checkpoint_path()
    path.write_text(path.read_text().replace(store.fingerprint, "0" * 24, 1))
    assert store.restore_checkpoint() is None


def test_non_semantic_settings_do_not_change_fingerprint(tmp_path):
    base = _store(tmp_path, config=make_config(use_cache=True, max_concurrency=8))
    other = _store(tmp_path, config=make_config(use_cache=False, max_concurrency=64))
    assert base.fingerprint == other.fingerprint


def test_kb_fingerprint_is_content_hash():
    kb = make_kb()
    assert kb_fingerprint(kb) == kb_fingerprint(make_kb())
    changed = make_kb()
    changed.items[0].text = "different"
    assert kb_fingerprint(changed) != kb_fingerprint(kb)
    reordered = KnowledgeBase(items=list(reversed(make_kb().items)))
    assert kb_fingerprint(reordered) == kb_fingerprint(kb)
