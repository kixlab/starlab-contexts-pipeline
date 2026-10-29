"""CLI: ledger-based budget analysis and saved-result commands (no model calls)."""

from __future__ import annotations

import csv
import json

import pytest

from pipeline import cli as cli_mod
from contextual.reporting import budget_rows, tau_sweep
from pipeline.cli import main
from contextual.models import RunResult
from pipeline.runner import PipelineRunner
from contextual.store import RunStore

from tests.pipeline.helpers import (
    ScriptedLabeler,
    ScriptedProposer,
    ScriptedVocabBuilder,
    TableEmbedder,
    label_book,
    make_config,
    make_facet,
    make_kb,
)
from tests.pipeline.test_experiments import spec_dict


@pytest.fixture
def saved_result(tmp_path):
    """Run: A accepted, A2 accepted (pool full), V vocab-failed, refusal, B replaces A."""
    script = [[make_facet("A")], [make_facet("A2")], [make_facet("V")], [],
              [make_facet("B")]]
    runner = PipelineRunner(
        config=make_config(proposal_budget=4, pool_size=2),
        task="t", knowledge_base=make_kb(),
        proposer=ScriptedProposer(script),
        vocab_builder=ScriptedVocabBuilder(fail_ids={"V"}),
        labeler=ScriptedLabeler(), embedder=TableEmbedder(),
        cache_dir=tmp_path / "cache",
    )
    runner.run()
    return runner.store.result_path()


def test_budget_rows_replay_ledger(saved_result):
    rows = budget_rows(RunStore.load_result(saved_result))
    assert [r["status"] for r in rows] == [
        "accepted", "accepted", "vocabulary_failed", "refused", "replaced",
    ]
    assert [r["budget_spent"] for r in rows] == [1, 2, 3, 3, 4]
    assert [r["tree_nodes"] for r in rows] == [2, 2, 2, 2, 3]
    assert [r["pool_size"] for r in rows] == [1, 2, 2, 2, 2]
    assert rows[4]["title"] == "B"


def test_budget_command_prints_table(saved_result, capsys):
    main(["budget", str(saved_result)])
    out = capsys.readouterr().out
    assert "Budget analysis" in out
    lines = [l for l in out.splitlines() if l.strip().startswith(("1 ", "2 ", "3 ", "4 ", "5 "))]
    assert len(lines) == 5
    assert " *B" in lines[4]  # tree grew when B replaced A
    assert "stop: proposal_budget" in out


def test_show_and_compare(saved_result, capsys):
    main(["show", str(saved_result)])
    out = capsys.readouterr().out
    assert "stop: proposal_budget" in out and "context: facets+history+items" in out
    assert "- B (global)" in out

    main(["compare", str(saved_result), str(saved_result)])
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 4 and "proposal_budget" in out[2]


def test_dump_writes_csv(saved_result, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli_mod, "STATIC_DIR", tmp_path / "static")
    main(["dump", str(saved_result)])
    rows = list(csv.reader(open(tmp_path / "static" / f"facets_{saved_result.stem}.csv")))
    assert rows[0] == ["Facet ID", "Details"]
    assert [r[0] for r in rows[1:]] == ["B", "A2"]
    assert "**Values:**" in rows[1][1]


def test_list_reports_legacy_files(tmp_path, capsys):
    (tmp_path / "a.json").write_text(json.dumps(spec_dict("a")))
    legacy = spec_dict("old")
    legacy["discovery"] = legacy.pop("pipeline")
    (tmp_path / "old.json").write_text(json.dumps(legacy))

    main(["list"], experiments_dir=tmp_path)
    out = capsys.readouterr().out
    assert "a " in out and "facets+items" in out
    assert "1 legacy-format file(s) skipped: old.json" in out


def test_unknown_target_exits(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps(spec_dict("a")))
    with pytest.raises(SystemExit):
        main(["show", "nope"], experiments_dir=tmp_path)


# ---- tau sweep and usage display ----

def _pool_result(fids, usage=None):
    return RunResult(
        pool=[make_facet(f) for f in fids], annotations=label_book(fids), tree=None,
        accepted_schemas=[], rejected_schemas=[], ledger=[], stop_reason=None,
        attempts=0, proposals=0, iterations=0, fingerprint="x", usage=usage or {},
    )


def test_tau_sweep_rebuilds_tree_per_threshold():
    rows = tau_sweep(_pool_result(["A", "A2", "B"]), make_kb(), make_config(), [0.2, 1.0])
    # tau 0.2: A2 is redundant with A -> root, A, B. tau 1.0: nothing exceeds it.
    assert [r["tree_nodes"] for r in rows] == [3, 4]
    assert [r["rejected_schemas"] for r in rows] == [1, 0]


def test_tree_command(monkeypatch, capsys, tmp_path):
    spec = cli_mod.ExperimentSpec.from_dict(spec_dict("exp"))
    monkeypatch.setattr(cli_mod, "resolve",
                        lambda target, d: ("exp", _pool_result(["A", "A2", "B"]), spec))
    monkeypatch.setattr(cli_mod, "cached_knowledge_base", lambda s: make_kb())
    main(["tree", "exp", "--tau", "0.2", "1.0"], experiments_dir=tmp_path)
    out = capsys.readouterr().out
    assert "Tree by refinement threshold: exp (3 pool facets)" in out
    assert " 0.20       3" in out and " 1.00       4" in out


def test_tree_command_needs_cached_kb(monkeypatch, tmp_path):
    spec = cli_mod.ExperimentSpec.from_dict(spec_dict("exp"))
    monkeypatch.setattr(cli_mod, "resolve", lambda t, d: ("exp", _pool_result(["A"]), spec))
    monkeypatch.setattr(cli_mod, "cached_knowledge_base", lambda s: None)
    with pytest.raises(SystemExit):
        main(["tree", "exp"], experiments_dir=tmp_path)


def test_show_reports_usage(saved_result, capsys):
    result = RunStore.load_result(saved_result)
    result.usage = {"proposal": {"input_tokens": 1500, "output_tokens": 10,
                                 "thinking_tokens": 7, "requests": 2}}
    print(cli_mod.format_summary("x", result))
    out = capsys.readouterr().out
    assert "proposal" in out and "1,500" in out and "total" in out
