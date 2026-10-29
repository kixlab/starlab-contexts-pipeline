"""Shared candidate lifecycle: validate -> dedup -> IRR -> vocabulary -> labeling."""

from __future__ import annotations

import math

import pytest

from pipeline import candidates as candidates_mod
from pipeline.candidates import CandidateFailure, CandidateProcessor, ProcessedCandidate
from pipeline.config import stage_seed
from pipeline.models import CandidateStatus

from tests.pipeline.helpers import (
    ScriptedLabeler,
    ScriptedVocabBuilder,
    TableEmbedder,
    embedding_of,
    make_config,
    make_facet,
    make_kb,
)


def _processor(embedder=None, vocab=None, labeler=None, **config_overrides):
    return CandidateProcessor(
        config=make_config(**config_overrides),
        task="test task",
        knowledge_base=make_kb(),
        vocab_builder=vocab or ScriptedVocabBuilder(),
        labeler=labeler or ScriptedLabeler(),
        embedder=embedder or TableEmbedder(),
    )


def _unit(deg: float) -> list[float]:
    return [math.cos(math.radians(deg)), math.sin(math.radians(deg))]


# ---- dedup ----

def test_duplicate_above_threshold():
    emb = TableEmbedder({"A": [1, 0], "A2": [1, 0]})
    proc = _processor(embedder=emb, dedup_similarity_threshold=0.9)
    existing = {"A": embedding_of(emb, make_facet("A"))}

    out = proc.process(make_facet("A2"), existing)

    assert isinstance(out, CandidateFailure)
    assert out.status == CandidateStatus.DUPLICATE
    assert out.duplicate_of == "A"
    assert out.nearest_id == "A"
    assert out.nearest_similarity == pytest.approx(1.0)


def test_duplicate_exactly_at_threshold():
    # cos([1,0], [3,4]) = 3/5 = 0.6 exactly
    emb = TableEmbedder({"A": [1, 0], "B": [3, 4]})
    proc = _processor(embedder=emb, dedup_similarity_threshold=0.6)
    existing = {"A": embedding_of(emb, make_facet("A"))}

    out = proc.process(make_facet("B"), existing)
    assert isinstance(out, CandidateFailure)
    assert out.status == CandidateStatus.DUPLICATE


def test_distinct_below_threshold_records_nearest_neighbor():
    emb = TableEmbedder({"A": [1, 0], "B": [4, 3]})  # cos = 0.8
    proc = _processor(embedder=emb, dedup_similarity_threshold=0.9)
    existing = {"A": embedding_of(emb, make_facet("A"))}

    out = proc.process(make_facet("B"), existing)

    assert isinstance(out, ProcessedCandidate)
    assert out.nearest_id == "A"
    assert out.nearest_similarity == pytest.approx(0.8)
    assert out.embedding  # retained for future dedup


def test_first_candidate_has_no_nearest_neighbor():
    out = _processor().process(make_facet("A"), {})
    assert isinstance(out, ProcessedCandidate)
    assert out.nearest_id is None
    assert out.nearest_similarity is None


def test_within_batch_dedup_follows_output_order():
    emb = TableEmbedder({"A": [1, 0], "A2": [1, 0]})
    proc = _processor(embedder=emb, dedup_similarity_threshold=0.9)

    outs = proc.process_batch([make_facet("A2"), make_facet("A")], {})

    assert isinstance(outs[0], ProcessedCandidate) and outs[0].facet.id == "A2"
    assert isinstance(outs[1], CandidateFailure)
    assert outs[1].status == CandidateStatus.DUPLICATE
    assert outs[1].duplicate_of == "A2"


def test_within_batch_dedup_does_not_chain_through_duplicates():
    # X=0deg, Y=18deg (cos .951 to X), Z=36deg (cos .951 to Y, .809 to X).
    emb = TableEmbedder({"X": _unit(0), "Y": _unit(18), "Z": _unit(36)})
    proc = _processor(embedder=emb, dedup_similarity_threshold=0.9)

    outs = proc.process_batch([make_facet("X"), make_facet("Y"), make_facet("Z")], {})

    assert isinstance(outs[0], ProcessedCandidate)
    assert outs[1].status == CandidateStatus.DUPLICATE and outs[1].duplicate_of == "X"
    # Z is only near the duplicate Y, which never entered the lifecycle.
    assert isinstance(outs[2], ProcessedCandidate)
    assert outs[2].nearest_id == "X"
    assert outs[2].nearest_similarity == pytest.approx(math.cos(math.radians(36)))


def test_dedup_text_excludes_guidelines_and_vocabulary():
    emb = TableEmbedder()
    f = make_facet("A", definition="what it captures")
    f.guidelines = ["SECRET-GUIDELINE"]
    _processor(embedder=emb).process(f, {})
    assert emb.texts == ["A (global): what it captures"]


# ---- validation ----

def test_invalid_candidate_is_not_embedded():
    emb = TableEmbedder()
    out = _processor(embedder=emb).process(make_facet("A", title="  "), {})
    assert isinstance(out, CandidateFailure)
    assert out.status == CandidateStatus.INVALID
    assert out.embedding is None
    assert emb.texts == []


# ---- IRR ----

def test_irr_pruning_removes_candidate(monkeypatch):
    calls = []

    def fake_irr(**kwargs):
        calls.append(kwargs)
        return {"low_irr_facet_ids": ["A"]}

    monkeypatch.setattr(candidates_mod, "compute_irr", fake_irr)
    vocab = ScriptedVocabBuilder()
    proc = _processor(vocab=vocab, irr_sample_videos=3)

    out = proc.process(make_facet("A"), {})

    assert isinstance(out, CandidateFailure)
    assert out.status == CandidateStatus.IRR_PRUNED
    assert out.embedding  # entered the lifecycle: kept for dedup
    assert vocab.calls == []  # regression vs legacy no-op: never reaches vocabulary
    assert calls[0]["sample_videos"] == 3


def test_irr_disabled_when_zero_sample_videos(monkeypatch):
    def boom(**kwargs):
        raise AssertionError("IRR should not run")

    monkeypatch.setattr(candidates_mod, "compute_irr", boom)
    out = _processor(irr_sample_videos=0).process(make_facet("A"), {})
    assert isinstance(out, ProcessedCandidate)


# ---- vocabulary / labeling ----

def test_vocabulary_failure():
    labeler = ScriptedLabeler()
    proc = _processor(vocab=ScriptedVocabBuilder(fail_ids={"A"}), labeler=labeler)
    out = proc.process(make_facet("A"), {})
    assert isinstance(out, CandidateFailure)
    assert out.status == CandidateStatus.VOCABULARY_FAILED
    assert labeler.calls == []


def test_labeling_failure():
    proc = _processor(labeler=ScriptedLabeler(fail_ids={"A"}))
    out = proc.process(make_facet("A"), {})
    assert isinstance(out, CandidateFailure)
    assert out.status == CandidateStatus.LABELING_FAILED


def test_processed_candidate_carries_annotations():
    out = _processor().process(make_facet("A"), {})
    assert isinstance(out, ProcessedCandidate)
    assert set(out.annotations.facet_ids()) == {"A"}


# ---- batching ----

def test_batch_issues_one_call_per_stage(monkeypatch):
    irr_calls = []

    def fake_irr(**kwargs):
        irr_calls.append([f.id for f in kwargs["facets"]])
        return {"low_irr_facet_ids": ["B"]}

    monkeypatch.setattr(candidates_mod, "compute_irr", fake_irr)
    emb = TableEmbedder({"A": [1, 0], "A2": [1, 0]})
    vocab = ScriptedVocabBuilder(fail_ids={"C"})
    labeler = ScriptedLabeler(fail_ids={"D"})
    proc = _processor(embedder=emb, vocab=vocab, labeler=labeler,
                      irr_sample_videos=3, dedup_similarity_threshold=0.9)

    facets = [make_facet(x) for x in ["A", "A2", "B", "C", "D"]]
    outs = proc.process_batch(facets, {})

    assert [o.status if isinstance(o, CandidateFailure) else "ok" for o in outs] == [
        "ok",
        CandidateStatus.DUPLICATE,
        CandidateStatus.IRR_PRUNED,
        CandidateStatus.VOCABULARY_FAILED,
        CandidateStatus.LABELING_FAILED,
    ]
    assert irr_calls == [["A", "B", "C", "D"]]
    assert vocab.calls == [["A", "C", "D"]]
    assert labeler.calls == [["A", "D"]]


# ---- sampling seeds ----

def test_stage_seeds_follow_run_seed_and_attempt(monkeypatch):
    seen = {}

    def fake_irr(**kwargs):
        seen["irr"] = kwargs["seed"]
        return {"low_irr_facet_ids": []}

    class SeedVocab(ScriptedVocabBuilder):
        def build_vocabulary(self, task, facets, knowledge_base, sample_videos=0, seed=None):
            seen["vocabulary"] = seed
            return super().build_vocabulary(task, facets, knowledge_base, sample_videos)

    class SeedLabeler(ScriptedLabeler):
        def label_batch(self, task, facets, knowledge_base, temperature=None, seed=None):
            seen["labeling"] = seed
            return super().label_batch(task, facets, knowledge_base)

    monkeypatch.setattr(candidates_mod, "compute_irr", fake_irr)
    proc = _processor(vocab=SeedVocab(), labeler=SeedLabeler(),
                      irr_sample_videos=3, seed=4)
    proc.process(make_facet("A"), {}, index=7)
    assert seen == {stage: stage_seed(4, stage, 7)
                    for stage in ("irr", "vocabulary", "labeling")}
    assert len(set(seen.values())) == 3
