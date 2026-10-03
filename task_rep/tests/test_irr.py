"""IRR: independent raters (distinct seeds, one batch) and kappa-based flagging."""

from __future__ import annotations

from typing import Dict, List, Optional

from facets import AnnotationTable, Facet, FacetLabel
from facets import label as label_mod
from facets.label import VLMFacetLabeler
from knowledge_items.schema import KnowledgeBase, KnowledgeItem, KnowledgeItemType, KnowledgeSource
from optimization.irr import IRR_BASE_SEED, IRR_TEMPERATURE, compute_irr
from utils import ModelConfig
from utils.llm import LLMRequestFailed


def _kb() -> KnowledgeBase:
    return KnowledgeBase(items=[
        KnowledgeItem(id=f"i{n}", type=KnowledgeItemType.ATOMIC, text=f"seg {n}",
                      source=KnowledgeSource(video_id="v1" if n <= 3 else "v2"),
                      unit_id=f"u{n}")
        for n in range(1, 7)
    ])


def _facet(fid: str) -> Facet:
    return Facet(id=fid, title=fid, title_plural=fid + "s", scope="global",
                 definition="d", justification="j", guidelines=[],
                 vocabulary=[FacetLabel(label="a", definition="a"),
                             FacetLabel(label="b", definition="b")])


class SeedLabeler:
    """Labels depend on the seed: ``flip`` facets disagree across raters."""

    def __init__(self, flip: set):
        self.flip = flip
        self.calls: List[Dict] = []

    def label_batch_multi(self, task, facets, knowledge_base, seeds, temperature=None):
        self.calls.append({"seeds": list(seeds), "temperature": temperature,
                           "videos": sorted(knowledge_base.by_video())})
        passes = []
        for seed in seeds:
            ann = AnnotationTable()
            for n, item in enumerate(knowledge_base):
                for f in facets:
                    label = "a" if n % 2 == 0 else "b"
                    if f.id in self.flip and (seed + n) % 2 == 0:
                        label = "b" if label == "a" else "a"
                    ann[(item.id, f.id)] = label
            passes.append(ann)
        return passes


def test_irr_runs_raters_in_one_batch_with_distinct_seeds():
    labeler = SeedLabeler(flip=set())
    compute_irr(task="t", facets=[_facet("f")], knowledge_base=_kb(), labeler=labeler,
                sample_videos=2, num_raters=3, min_agreement=0.8)
    assert len(labeler.calls) == 1
    call = labeler.calls[0]
    assert call["seeds"] == [IRR_BASE_SEED, IRR_BASE_SEED + 1, IRR_BASE_SEED + 2]
    assert call["temperature"] is IRR_TEMPERATURE is None
    assert call["videos"] == ["v1", "v2"]


def test_irr_flags_only_facets_raters_disagree_on():
    result = compute_irr(task="t", facets=[_facet("stable"), _facet("noisy")],
                         knowledge_base=_kb(), labeler=SeedLabeler(flip={"noisy"}),
                         sample_videos=2, num_raters=3, min_agreement=0.8)
    assert result["per_facet_kappa"]["stable"] == 1.0
    assert result["per_facet_kappa"]["noisy"] < 0.8
    assert result["low_irr_facet_ids"] == ["noisy"]


def test_label_batch_multi_groups_requests_by_seed(monkeypatch):
    seen = {}

    def fake_batch(batch_id, request_args, func_request, func_response, *,
                   max_concurrency, model_config, temperature=None, seeds=None,
                   allow_errors=False):
        seen["seeds"] = seeds
        seen["temperature"] = temperature
        # Label every item with the request's seed, so passes are distinguishable.
        return [
            [{"item_id": item.id, "label": f"s{seed}"} for item in args["video_items"]]
            for args, seed in zip(request_args, seeds)
        ]

    monkeypatch.setattr(label_mod, "batch_run_llm_calls", fake_batch)
    labeler = VLMFacetLabeler(model_config=ModelConfig("vlm", "gemini", "x"))
    passes = labeler.label_batch_multi("t", [_facet("f"), _facet("g")], _kb(),
                                       seeds=[7, 8], temperature=None)

    # 2 seeds x 2 facets x 2 videos, all in one batch
    assert seen["seeds"] == [7, 7, 7, 7, 8, 8, 8, 8]
    assert len(passes) == 2
    assert set(passes[0].values()) == {"s7"} and set(passes[1].values()) == {"s8"}
    assert len(passes[0]) == 6 * 2  # every item under both facets

    single = labeler.label_batch("t", [_facet("f")], _kb())
    assert seen["seeds"] == [None, None]  # provider default seed
    assert set(single.values()) == {"sNone"}


# ---- permanent request failures ----

def _failing_batch(fail_when):
    """Fake batch runner: requests matching ``fail_when(args, seed)`` fail."""
    def fake_batch(batch_id, request_args, func_request, func_response, *,
                   max_concurrency, model_config, temperature=None, seeds=None,
                   allow_errors=False):
        assert allow_errors
        out = []
        for args, seed in zip(request_args, seeds):
            if fail_when(args, seed):
                out.append(LLMRequestFailed(errors=["boom"], kind="request"))
            else:
                out.append([{"item_id": i.id, "label": "a"} for i in args["video_items"]])
        return out
    return fake_batch


def _first_video(args):
    return args["video_items"][0].source.video_id == "v1"


def test_label_batch_drops_facet_with_any_failed_request(monkeypatch):
    monkeypatch.setattr(label_mod, "batch_run_llm_calls", _failing_batch(
        lambda args, seed: args["facet"].id == "f" and _first_video(args)))
    labeler = VLMFacetLabeler(model_config=ModelConfig("vlm", "gemini", "x"))
    table = labeler.label_batch("t", [_facet("f"), _facet("g")], _kb())
    assert set(table.facet_ids()) == {"g"}  # f is not partially labeled


def test_label_batch_multi_keeps_partial_passes(monkeypatch):
    monkeypatch.setattr(label_mod, "batch_run_llm_calls", _failing_batch(
        lambda args, seed: seed == 2 and _first_video(args)))
    labeler = VLMFacetLabeler(model_config=ModelConfig("vlm", "gemini", "x"))
    p1, p2 = labeler.label_batch_multi("t", [_facet("f")], _kb(), seeds=[1, 2])
    assert len(p1) == 6 and len(p2) == 3  # pass 2 lost only v1's items


class PartialLabeler(SeedLabeler):
    """Rater 0 is missing labels for the first ``missing`` items."""

    def __init__(self, missing):
        super().__init__(flip=set())
        self.missing = missing

    def label_batch_multi(self, task, facets, knowledge_base, seeds, temperature=None):
        passes = super().label_batch_multi(task, facets, knowledge_base, seeds, temperature)
        missing_ids = {item.id for item in list(knowledge_base)[: self.missing]}
        passes[0] = AnnotationTable(
            {key: label for key, label in passes[0].items() if key[0] not in missing_ids}
        )
        return passes


def test_irr_ignores_items_missing_a_rater():
    result = compute_irr(task="t", facets=[_facet("f")], knowledge_base=_kb(),
                         labeler=PartialLabeler(missing=2), sample_videos=2, num_raters=3)
    assert result["per_facet_kappa"]["f"] == 1.0  # gaps are not disagreements


def test_irr_with_no_fully_rated_items_flags_facet():
    result = compute_irr(task="t", facets=[_facet("f")], knowledge_base=_kb(),
                         labeler=PartialLabeler(missing=6), sample_videos=2, num_raters=3,
                         min_agreement=0.8)
    assert result["low_irr_facet_ids"] == ["f"]
