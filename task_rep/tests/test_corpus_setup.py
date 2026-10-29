"""Corpus registration and saved-run export need neither models nor downloads."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from export_bundle import main as export_main

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("register_corpus", ROOT / "scripts/register_corpus.py")
registration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(registration)


def test_registration_creates_a_local_task_and_experiment_without_copying_frame_paths(tmp_path):
    for rel in ["task_rep/data/tasks.json", "task_rep/experiments/paper_folding_demo.json"]:
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text((ROOT / rel).read_text())
    source = tmp_path / "transcripts.json"
    source.write_text(json.dumps([{"id": "v", "title": "Tutorial", "url": "", "frames": [{"path": "private-frame"}],
                                   "transcript": [{"text": "Prepare", "start": 0, "end": 5}]}]))
    dataset, experiment = registration.register_corpus(source, "local-corpus", "My corpus", root=tmp_path)
    assert json.loads(dataset.read_text())[0]["frames"] == []
    config = json.loads(experiment.read_text())
    assert config["name"] == config["task_id"] == "local-corpus"
    assert not config["constructor"]["use_vision"]
    tasks = json.loads((tmp_path / "task_rep/data/local-tasks.json").read_text())
    assert tasks["local-corpus"]["video_ids"] == ["v"]
    with pytest.raises(ValueError, match="already exists"):
        registration.register_corpus(source, "local-corpus", "My corpus", root=tmp_path)


@pytest.mark.parametrize("videos", [[], [{"id": "v", "transcript": []}],
    [{"id": "v", "transcript": [{"text": "Prepare", "start": 5, "end": 0}]}]])
def test_invalid_corpora_are_rejected_before_registration(videos):
    with pytest.raises(ValueError):
        registration.prepare_corpus(videos)


def test_run_folder_export_resolves_recorded_units(tmp_path, monkeypatch):
    root = tmp_path / "cache/task"
    run = root / "runs/saved"
    run.mkdir(parents=True)
    units = root / "units/recorded.jsonl"
    units.parent.mkdir()
    units.write_text(json.dumps({"id": "p", "type": "step", "text": "Prepare", "source": {"video_id": "v", "time_range": [0, 5]}}) + "\n")
    (run / "run.json").write_text(json.dumps({"sources": {"units": "units/recorded.jsonl"}}))
    (run / "result.json").write_text(json.dumps({"pool": [{"id": "phase", "title": "Phase", "vocabulary": [{"label": "prepare"}]}],
                                               "annotations": [["p", "phase", "prepare"]]}))
    output = tmp_path / "bundle.json"
    monkeypatch.setattr(sys, "argv", ["export", str(run), "--output", str(output)])
    export_main()
    assert json.loads(output.read_text())["labeled_dataset"][0]["pieces"][0]["labels"] == {"phase": ["prepare"]}


def test_experiment_export_finds_run_units_and_original_metadata(tmp_path, monkeypatch):
    from pipeline import experiments
    from config import Config
    from types import SimpleNamespace

    root = tmp_path / "cache/task"
    run = root / "runs/saved"
    run.mkdir(parents=True)
    units = root / "units/recorded.jsonl"
    units.parent.mkdir()
    units.write_text(json.dumps({"id": "p", "type": "step", "text": "Prepare", "source": {"video_id": "v", "time_range": [0, 5]}}) + "\n")
    (run / "run.json").write_text(json.dumps({"sources": {"units": "units/recorded.jsonl"}}))
    (run / "result.json").write_text(json.dumps({"pool": [], "annotations": []}))
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    (datasets / "local-corpus.json").write_text(json.dumps([{"id": "v", "title": "Original tutorial", "url": "/media/source.mp4"}]))
    monkeypatch.setattr(Config, "TASK_DATASETS_DIR", datasets)
    monkeypatch.setattr(experiments, "load_experiments_dir", lambda: ({"local-corpus": SimpleNamespace(task_id="local-corpus")}, []))
    monkeypatch.setattr(experiments, "find_result", lambda spec: run)
    output = tmp_path / "bundle.json"
    monkeypatch.setattr(sys, "argv", ["export", "--experiment", "local-corpus", "--output", str(output)])
    export_main()
    assert json.loads(output.read_text())["labeled_dataset"][0]["title"] == "Original tutorial"
