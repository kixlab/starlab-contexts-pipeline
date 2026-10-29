"""Corpus selection: a fixed video sample, independent of the run seed."""

from __future__ import annotations

from knowledge_items.schema import KnowledgeBase, KnowledgeItem, KnowledgeItemType, KnowledgeSource

from contextual.corpus import select_corpus
from contextual.reporting import tau_sweep
from contextual.models import RunResult

from tests.contextual.helpers import label_book, make_facet, make_kb, make_settings


def _wide_kb(n_videos=6) -> KnowledgeBase:
    return KnowledgeBase(items=[
        KnowledgeItem(id=f"v{v}i{i}", type=KnowledgeItemType.ATOMIC, text=f"seg {v}-{i}",
                      source=KnowledgeSource(video_id=f"v{v}"), unit_id=f"u{v}{i}")
        for v in range(n_videos) for i in range(2)
    ])


def _videos(kb):
    return sorted(kb.by_video())


def test_default_keeps_everything():
    kb = _wide_kb()
    assert select_corpus(kb, make_settings()) is kb
    assert select_corpus(kb, make_settings(corpus_sample_videos=10)) is kb


def test_sample_is_fixed_by_corpus_seed_not_run_seed():
    kb = _wide_kb()
    a = _videos(select_corpus(kb, make_settings(corpus_sample_videos=3, seed=0)))
    b = _videos(select_corpus(kb, make_settings(corpus_sample_videos=3, seed=5)))
    c = _videos(select_corpus(kb, make_settings(corpus_sample_videos=3, corpus_seed=1)))
    assert len(a) == 3 and a == b and a != c
    sub = select_corpus(kb, make_settings(corpus_sample_videos=3))
    assert all(len(items) == 2 for items in sub.by_video().values())  # whole videos


def test_tau_sweep_uses_the_runs_corpus():
    # D copies B inside the sampled video and is independent of B in the other:
    # distinct on the full corpus (MI/H = 0.19 < tau), redundant on the sample.
    settings = make_settings(corpus_sample_videos=1)
    sampled = next(iter(select_corpus(make_kb(), settings).by_video()))
    b = "aabbaabb"
    d = "aabbabab" if sampled == "v1" else "ababaabb"
    result = RunResult(pool=[make_facet("B"), make_facet("D")],
                       annotations=label_book(["B", "D"], {"B": b, "D": d}), tree=None,
                       accepted_schemas=[], rejected_schemas=[], ledger=[], stop_reason=None,
                       attempts=0, proposals=0, iterations=0, fingerprint="x")
    full = tau_sweep(result, make_kb(), make_settings(), [0.2])[0]
    sample = tau_sweep(result, make_kb(), settings, [0.2])[0]
    assert (full["tree_nodes"], full["rejected_schemas"]) == (3, 0)
    assert (sample["tree_nodes"], sample["rejected_schemas"]) == (2, 1)
