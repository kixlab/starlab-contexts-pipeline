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


# ---- reports.transcripts ----

def test_outline_splits_headings_videos_and_factors():
    from facets.prompts.fa_facet_iterative import SECTION_SCHEMAS
    from reports.transcripts import outline

    text = "\n".join([
        "Opening line.",
        "TUTORIALS",
        "Every segment, grouped by video.",
        "Video v1:", '<segment id="S1">', "a", "</segment>",
        "Video v2:", '<segment id="S2">', "b", "</segment>",
        "- Contextual factors in the pool:",
        '<factor id="F1">', "<title>Roll Type</title>", "</factor>",
        SECTION_SCHEMAS.split("\n", 1)[0],
        "Design space: kept 1 of 1 single factors and 0 of 0 pairs.",
        "", "Kept combinations:", "- C1: F1 (Roll Type), 2 cells",
        "", "Pruned single factors:", "- none",
    ])
    sections = outline(text)
    assert [s["title"] for s in sections] == [
        None, "TUTORIALS", "Contextual factors in the pool",
        "The current design space of factor combinations",
        "Design space: kept 1 of 1 single factors and 0 of 0 pairs."]
    tutorials, pool, heading, space = sections[1:]
    assert tutorials["text"] == "Every segment, grouped by video."
    assert [c["title"] for c in tutorials["children"]] == ["Video v1 · 1 segments",
                                                           "Video v2 · 1 segments"]
    assert [c["title"] for c in pool["children"]] == ["F1 · Roll Type"]
    assert heading["text"].startswith("A pruned combination does not make")
    assert [c["title"] for c in space["children"]] == ["Kept combinations (1)",
                                                       "Pruned single factors"]
    # headings become titles; every other line stays in a section or child text
    texts = [t for s in sections for t in [s["text"]] + [c["text"] for c in s.get("children", [])]]
    for line in ("Opening line.", "a", "b", "<title>Roll Type</title>",
                 "- C1: F1 (Roll Type), 2 cells", "- none"):
        assert any(line in t.split("\n") for t in texts), line
