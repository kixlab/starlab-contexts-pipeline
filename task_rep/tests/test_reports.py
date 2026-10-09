"""Tests for analysis.rescore and the reports package (no model calls)."""

from __future__ import annotations

import pytest

from analysis.rescore import rescore_evaluation
from contextual.models import CandidateRecord, CandidateStatus, RunState, TreeQualitySnapshot
from reports.pilot_page import render
from reports.runs import parse_overrides

from tests.contextual.helpers import make_settings
from tests.test_measures import setup  # noqa: F401  (fixture)


def _state(facets, ann):
    return RunState(
        pool=list(facets), annotations=ann,
        ledger=[
            CandidateRecord(attempt_index=1, status=CandidateStatus.ACCEPTED,
                            facet_id=facets[0].id,
                            tree_quality=TreeQualitySnapshot(0, 0, [facets[0].id])),
            CandidateRecord(attempt_index=2, status=CandidateStatus.ACCEPTED,
                            facet_id=facets[1].id,
                            tree_quality=TreeQualitySnapshot(0, 0, [f.id for f in facets])),
        ],
        runner_state={"rescored": {"date": "earlier"}},
    )


def test_rescore_reevaluates_tree_and_snapshots(setup):  # noqa: F811
    kb, f, ann = setup
    state = _state([f["video"], f["video_copy"]], ann)
    rescore_evaluation(state, kb, make_settings(refinement_threshold=0.55), {"date": "now"})
    # the renamed copy is redundant: one single kept (plus the root)
    assert [r.tree_quality.tree_nodes for r in state.ledger] == [2, 2]
    assert [s.facet_ids for s in state.rejected_schemas] == [["video_copy"]]
    assert state.runner_state["rescored"] == {"date": "now", "previous": {"date": "earlier"}}


def test_rescore_refuses_a_pool_that_changed(setup):  # noqa: F811
    kb, f, ann = setup
    state = _state([f["video"], f["position"]], ann)
    state.pool = [f["position"]]  # "video" was replaced out during the run
    with pytest.raises(ValueError, match="cannot be re-scored"):
        rescore_evaluation(state, kb, make_settings(), {})


def test_parse_overrides_reads_json_values():
    assert parse_overrides(["refinement_threshold=0.55", "max_schema_facets=2",
                            "metric_mode=entropy"]) == {
        "refinement_threshold": 0.55, "max_schema_facets": 2, "metric_mode": "entropy"}


def test_render_fills_placeholders_and_escapes():
    page = render({"title": "A <b> page", "eyebrow": "e", "intro": ["**Bold** & more"]},
                  {"versions": [], "note": "</script>"})
    assert "__TITLE__" not in page and "__DATA__" not in page and "__INTRO__" not in page
    assert "A &lt;b&gt; page" in page
    assert '<p class="lede soft"><strong>Bold</strong> &amp; more</p>' in page
    assert "<\\/script>" in page  # data cannot close the script tag
