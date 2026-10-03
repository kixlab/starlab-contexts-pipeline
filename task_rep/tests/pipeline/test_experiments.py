"""Experiment adapter: spec loading, KB construction hand-off, result lookup, summary."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass

import pytest

from knowledge_items import KnowledgeItemType

from pipeline import experiments as experiments_mod
from pipeline.config import A2_CONTEXT, ProposalStrategy
from contextual.reporting import summarize
from pipeline.experiments import (
    EXPERIMENTS_DIR,
    ExperimentSpec,
    LegacyExperimentError,
    Services,
    find_result,
    load_experiments_dir,
    run_experiment,
)

from tests.pipeline.helpers import (
    ScriptedLabeler,
    ScriptedProposer,
    ScriptedVocabBuilder,
    TableEmbedder,
    make_facet,
    make_kb,
)

# Legacy collision/pareto configs, deleted in migration step 9.
OBSOLETE = {
    "photo_editing_step_tree", "sushi_roll_step_tree", "test_makeup_step",
    "test_photo_editing_step", "test_sushi_roll_3_step", "test_sushi_roll_step",
    "test_caramel_apple_atomic", "test_caramel_apple_step",
    "test_caramel_apple_tutorial", "test_makeup_atomic", "test_photo_editing_atomic",
    "test_remove_object_atomic", "test_sushi_roll_3_atomic",
    "test_sushi_roll_3_atomic_gpt5", "test_sushi_roll_3_tutorial_gpt5",
    "test_sushi_roll_atomic", "test_sushi_roll_tutorial",
}

VLM = {"category": "vlm", "provider": "gemini", "model": "gemini-3-flash-preview"}
EMB = {"category": "embedding", "provider": "gemini", "model": "gemini-embedding-2"}


def spec_dict(name="exp", **pipeline_overrides):
    pipeline = {
        "item_scope": {"item_type": "atomic", "atomic_subtype": "all"},
        "proposal_model": VLM, "vocab_model": VLM, "labeling_model": VLM,
        "strategy": "iterative", "context": ["facets", "items"],
        "proposal_budget": 4, "pool_size": 4, "irr_sample_videos": 0,
        "use_cache": False,
    }
    pipeline.update(pipeline_overrides)
    return {
        "name": name, "task_id": "sushi_roll", "description": "d", "tags": ["t"],
        "constructor": {
            "use_vision": True, "use_cache": True,
            "cluster_config": {"similarity_threshold": 0.85, "model_config": EMB},
            "model_config": VLM,
        },
        "pipeline": pipeline,
    }


# ---- spec loading ----

def test_embedding_model_comes_from_cluster_config():
    spec = ExperimentSpec.from_dict(spec_dict())
    assert spec.pipeline.embedding_model.to_dict() == EMB
    assert spec.item_type == KnowledgeItemType.ATOMIC


def test_spec_round_trip():
    data = spec_dict(context=["schemas"], strategy="batch")
    spec = ExperimentSpec.from_dict(data)
    again = ExperimentSpec.from_dict(spec.to_dict())
    assert again.to_dict() == spec.to_dict()
    assert again.pipeline.context == A2_CONTEXT
    assert again.pipeline.strategy == ProposalStrategy.BATCH
    assert "embedding_model" not in spec.to_dict()["pipeline"]


def test_legacy_discovery_block_rejected():
    data = spec_dict()
    data["discovery"] = data.pop("pipeline")
    with pytest.raises(LegacyExperimentError):
        ExperimentSpec.from_dict(data)


def test_explicit_embedding_model_rejected():
    with pytest.raises(ValueError):
        ExperimentSpec.from_dict(spec_dict(embedding_model=EMB))


def test_unknown_keys_rejected():
    data = spec_dict()
    data["extra"] = 1
    with pytest.raises(ValueError):
        ExperimentSpec.from_dict(data)
    with pytest.raises(ValueError):
        ExperimentSpec.from_dict(spec_dict(max_iterations=5))


def test_load_dir_skips_legacy_and_rejects_duplicates(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps(spec_dict("a")))
    legacy = spec_dict("old")
    legacy["discovery"] = legacy.pop("pipeline")
    (tmp_path / "old.json").write_text(json.dumps(legacy))

    specs, skipped = load_experiments_dir(tmp_path)
    assert list(specs) == ["a"]
    assert [p.name for p in skipped] == ["old.json"]

    (tmp_path / "b.json").write_text(json.dumps(spec_dict("a")))
    with pytest.raises(ValueError):
        load_experiments_dir(tmp_path)


def test_repository_experiments_are_migrated():
    specs, legacy = load_experiments_dir(EXPERIMENTS_DIR)
    assert {p.stem for p in legacy} <= OBSOLETE, "unmigrated experiment files"
    assert specs, "no experiments loaded"


# ---- running through the constructor hand-off ----

@dataclass
class FakeTask:
    name: str


def _services(script):
    return Services(
        proposer=ScriptedProposer(script),
        vocab_builder=ScriptedVocabBuilder(),
        labeler=ScriptedLabeler(),
        embedder=TableEmbedder(),
    )


def test_run_experiment_uses_constructor_kb(tmp_path, monkeypatch):
    calls = []

    def fake_load(task_id, constructor_config, item_type):
        calls.append((task_id, item_type, constructor_config.cluster_config.similarity_threshold))
        return FakeTask("sushi roll"), make_kb(), tmp_path

    monkeypatch.setattr(experiments_mod, "load_knowledge_base", fake_load)
    spec = ExperimentSpec.from_dict(spec_dict(proposal_budget=2, pool_size=2))
    result = run_experiment(spec, services=_services([[make_facet("A")], [make_facet("B")]]))

    assert calls == [("sushi_roll", KnowledgeItemType.ATOMIC, 0.85)]
    assert [f.id for f in result.pool] == ["A", "B"]
    assert result.config == spec.pipeline.semantic_dict()
    assert result.num_units == 8

    found = find_result(spec, tmp_path)
    assert found is not None and found.name.startswith("pipeline_result_")


def test_find_result_matches_config_only(tmp_path, monkeypatch):
    monkeypatch.setattr(experiments_mod, "load_knowledge_base",
                        lambda *a: (FakeTask("t"), make_kb(), tmp_path))
    run_experiment(ExperimentSpec.from_dict(spec_dict(proposal_budget=1, pool_size=1)),
                   services=_services([[make_facet("A")]]))

    other = ExperimentSpec.from_dict(spec_dict(proposal_budget=2, pool_size=2))
    assert find_result(other, tmp_path) is None


def test_find_result_fills_defaults_for_fields_added_later(tmp_path, monkeypatch):
    monkeypatch.setattr(experiments_mod, "load_knowledge_base",
                        lambda *a: (FakeTask("t"), make_kb(), tmp_path))
    spec = ExperimentSpec.from_dict(spec_dict(proposal_budget=1, pool_size=1))
    run_experiment(spec, services=_services([[make_facet("A")]]))

    # Simulate a result saved before `items_sample_videos` existed.
    path = find_result(spec, tmp_path)
    data = json.loads(path.read_text())
    del data["config"]["items_sample_videos"]
    path.write_text(json.dumps(data))
    assert find_result(spec, tmp_path) == path


def test_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(experiments_mod, "load_knowledge_base",
                        lambda *a: (FakeTask("t"), make_kb(), tmp_path))
    spec = ExperimentSpec.from_dict(spec_dict(proposal_budget=2, pool_size=2))
    result = run_experiment(spec, services=_services(
        [[], [make_facet("A")], [make_facet("B")]]
    ))
    s = summarize(result).to_dict()
    assert s == {
        "num_units": 8, "pool_facets": 2, "iterations": 3, "attempts": 3,
        "proposals": 2, "tree_nodes": 3, "tree_depth": s["tree_depth"],
        "stop_reason": "proposal_budget",
        "status_counts": {"refused": 1, "accepted": 2},
    }
