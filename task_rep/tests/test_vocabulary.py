"""Vocabulary building: tolerated per-video gaps, failed consolidation."""

from __future__ import annotations

from facets import Facet, FacetLabel
from facets import vocabulary as vocab_mod
from facets.vocabulary import VLMFacetVocabularyBuilder
from knowledge_items.schema import KnowledgeBase, KnowledgeItem, KnowledgeItemType, KnowledgeSource
from utils import ModelConfig
from utils.llm import LLMRequestFailed


def _kb() -> KnowledgeBase:
    return KnowledgeBase(items=[
        KnowledgeItem(id=f"i{n}", type=KnowledgeItemType.ATOMIC, text=f"seg {n}",
                      source=KnowledgeSource(video_id=f"v{n}"), unit_id=f"u{n}")
        for n in range(1, 4)
    ])


def _facet(fid: str) -> Facet:
    return Facet(id=fid, title=fid, title_plural=fid + "s", scope="global",
                 definition="d", justification="j", guidelines=[],
                 vocabulary=[FacetLabel(label="init", definition="initial")])


def _fake_batch(fail_proposal, fail_consolidation):
    """Proposal requests failing per (facet, video); consolidation per facet."""
    calls = []

    def fake(batch_id, request_args, func_request, func_response, *,
             max_concurrency, model_config, temperature=None, seeds=None,
             allow_errors=False):
        assert allow_errors
        calls.append(batch_id.split("_")[1])
        out = []
        for args in request_args:
            fid = args["facet"].id
            if "video_items" in args:  # proposal
                vid = args["video_items"][0].source.video_id
                out.append(LLMRequestFailed(["x"], "request") if (fid, vid) in fail_proposal
                           else [FacetLabel(label=f"{vid}", definition="d")])
            else:  # consolidation
                out.append(LLMRequestFailed(["x"], "request") if fid in fail_consolidation
                           else Facet.from_dict({**args["facet"].to_dict(), "vocabulary": [
                               {"label": "merged", "definition": "d"}]}))
        return out
    return fake


def _build(monkeypatch, fail_proposal=(), fail_consolidation=()):
    monkeypatch.setattr(vocab_mod, "batch_run_llm_calls",
                        _fake_batch(set(fail_proposal), set(fail_consolidation)))
    builder = VLMFacetVocabularyBuilder(model_config=ModelConfig("vlm", "gemini", "x"))
    return {f.id: f for f in builder.build_vocabulary("t", [_facet("f"), _facet("g")], _kb())}


def test_partial_proposal_failures_are_tolerated_and_noted(monkeypatch):
    out = _build(monkeypatch, fail_proposal={("f", "v1"), ("f", "v2")})
    assert [l.label for l in out["f"].vocabulary] == ["merged"]
    assert out["f"].metadata["vocab_proposal_failures"] == 2
    assert "vocab_proposal_failures" not in out["g"].metadata


def test_all_proposals_failed_means_no_vocabulary(monkeypatch):
    out = _build(monkeypatch, fail_proposal={("f", v) for v in ("v1", "v2", "v3")})
    assert out["f"].vocabulary == []  # -> vocabulary_failed in the pipeline
    assert [l.label for l in out["g"].vocabulary] == ["merged"]


def test_failed_consolidation_means_no_vocabulary(monkeypatch):
    out = _build(monkeypatch, fail_consolidation={"f"})
    assert out["f"].vocabulary == []
    assert [l.label for l in out["g"].vocabulary] == ["merged"]
