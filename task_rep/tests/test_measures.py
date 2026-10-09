"""Tests for the post-hoc measures in ``analysis.measures``."""

from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np
import pytest

from analysis.measures import (
    applicable_entropy,
    embedding_dispersion,
    information_matrices,
    junk_control,
    junk_facets,
    levels,
    partition_diversity,
    run_measures,
    shuffle_labels,
    unit_coverage,
    vendi,
)
from contextual.models import CandidateRecord, CandidateStatus, TreeQualitySnapshot
from facets import AnnotationTable, Facet, FacetLabel
from knowledge_items.schema import (
    KnowledgeBase, KnowledgeItem, KnowledgeItemType, KnowledgeSource,
)

from tests.contextual.helpers import make_settings


def _facet(fid, labels):
    return Facet(id=fid, title=fid, title_plural=fid, scope="global", definition=fid,
                 justification="", guidelines=[],
                 vocabulary=[FacetLabel(label=l, definition=l) for l in labels])


@pytest.fixture
def setup():
    """Two videos x three segments, one unit per segment.

    video: per-video value; video_copy: the same partition renamed;
    position: segment position within the video (independent of video);
    moment: applies only to each video's first segment.
    """
    items, ann = [], AnnotationTable()
    for v, vid in enumerate(["v1", "v2"]):
        for pos in range(3):
            iid = f"{vid}_{pos}"
            items.append(KnowledgeItem(id=iid, type=KnowledgeItemType.ATOMIC, text=iid,
                                       source=KnowledgeSource(video_id=vid), unit_id=f"u_{iid}"))
            ann[(iid, "video")] = ["x", "y"][v]
            ann[(iid, "video_copy")] = ["left", "right"][v]
            ann[(iid, "position")] = ["p", "q", "r"][pos]
            ann[(iid, "moment")] = ("wet" if v == 0 else "rice") if pos == 0 else "not applicable"
    facets = {
        "video": _facet("video", ["x", "y"]),
        "video_copy": _facet("video_copy", ["left", "right"]),
        "position": _facet("position", ["p", "q", "r"]),
        "moment": _facet("moment", ["wet", "rice", "not applicable"]),
    }
    return KnowledgeBase(items=items), facets, ann


def test_vendi_bounds():
    assert vendi(np.eye(4)) == pytest.approx(4.0)
    assert vendi(np.ones((4, 4))) == pytest.approx(1.0)


def test_nmi_identical_and_independent_partitions(setup):
    kb, f, ann = setup
    nmi, nvi, h = information_matrices(kb, [f["video"], f["video_copy"], f["position"]], ann)
    assert nmi[0, 1] == pytest.approx(1.0) and nvi[0, 1] == pytest.approx(0.0)
    assert nmi[0, 2] == pytest.approx(0.0, abs=1e-9) and nvi[0, 2] == pytest.approx(1.0)
    assert h[0] == pytest.approx(1.0) and h[2] == pytest.approx(math.log2(3))


def test_partition_diversity_counts_distinct_partitions(setup):
    kb, f, ann = setup
    same = partition_diversity(kb, [f["video"], f["video_copy"]], ann)
    different = partition_diversity(kb, [f["video"], f["position"]], ann)
    assert same["vendi_nmi"] == pytest.approx(1.0)
    assert different["vendi_nmi"] == pytest.approx(2.0)


def test_levels_report_singles_and_pairs_separately(setup):
    kb, f, ann = setup
    settings = make_settings(max_schema_facets=2, refinement_threshold=0.7)
    lv = levels(kb, [f["video"], f["video_copy"], f["position"]], ann, settings)
    # the renamed copy is pruned; video x position is an independent pair (position
    # alone carries log2(3)/log2(6) = 61% of it, under the 70% limit)
    assert lv["kept_singles"] == 2 and lv["pruned_singles"] == 1
    assert ["position", "video"] in [sorted(p) for p in lv["kept_pair_ids"]]


def test_shuffle_keeps_each_factors_value_distribution(setup):
    kb, f, ann = setup
    facets = [f["video"], f["moment"]]
    shuffled = shuffle_labels(kb, facets, ann, random.Random(1))
    for facet in facets:
        assert Counter(shuffled.by_facet(facet.id).values()) == \
            Counter(ann.by_facet(facet.id).values())


def test_junk_factors_are_consistent(setup):
    kb, _, _ = setup
    junk, ann = junk_facets(kb, seed=3)
    assert [j.id for j in junk] == ["junk_video_2", "junk_video_3", "junk_segment_4"]
    for vid in ("v1", "v2"):
        assert len({ann[(f"{vid}_{p}", "junk_video_3")] for p in range(3)}) == 1
    assert len(ann) == 3 * len(kb)


def test_junk_control_reports_surviving_junk(setup):
    kb, f, ann = setup
    out = junk_control(kb, [f["video"]], ann, make_settings(refinement_threshold=0.55))
    assert 0 <= out["junk_kept_singles"] <= 3 and "vendi_nmi" in out


def test_coverage_and_granularity_use_applicable_segments(setup):
    kb, f, ann = setup
    assert unit_coverage(kb, f["moment"], ann) == pytest.approx(2 / 6)
    assert unit_coverage(kb, f["video"], ann) == pytest.approx(1.0)
    # over its two applicable segments the moment factor splits 50/50
    assert applicable_entropy(kb, f["moment"], ann) == pytest.approx(1.0)


def test_embedding_dispersion():
    same = embedding_dispersion([[1.0, 0.0], [1.0, 0.0]])
    orthogonal = embedding_dispersion([[1.0, 0.0], [0.0, 1.0]])
    assert same["mean_cosine_distance"] == pytest.approx(0.0)
    assert same["vendi_cosine"] == pytest.approx(1.0)
    assert orthogonal["mean_cosine_distance"] == pytest.approx(1.0)
    assert orthogonal["vendi_cosine"] == pytest.approx(2.0)


def test_run_measures_saturation_follows_the_ledger(setup):
    kb, f, ann = setup
    pool = [f["video"], f["position"]]
    ledger = [
        CandidateRecord(attempt_index=1, status=CandidateStatus.ACCEPTED, facet_id="video",
                        tree_quality=TreeQualitySnapshot(2, 1, ["video"])),
        CandidateRecord(attempt_index=2, status=CandidateStatus.IRR_PRUNED, facet_id="zz",
                        tree_quality=TreeQualitySnapshot(2, 1, ["video"])),
        CandidateRecord(attempt_index=3, status=CandidateStatus.ACCEPTED, facet_id="position",
                        tree_quality=TreeQualitySnapshot(3, 2, ["video", "position"])),
    ]
    out = run_measures(kb, pool, ann, ledger, make_settings(max_schema_facets=2,
                       refinement_threshold=0.7), null_repeats=3)
    assert [(r["attempt"], r["accepted"], r["pool"]) for r in out["saturation"]] == \
        [(1, 1, 1), (2, 1, 1), (3, 2, 2)]
    assert out["at_run_tau"]["kept_pairs"] == 1
    assert out["shuffled_null"]["vendi_nmi"]["mean"] >= 1.0
    assert out["embeddings_found"] == 0
