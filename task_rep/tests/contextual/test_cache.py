"""contextual.cache: items/units reuse, run folders, append-only checkpoints."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path

import construct as construct_mod
from config import Config
from construct import ConstructorConfig
from contextual import cache
from contextual.evaluation import PoolEvaluator
from contextual.lifecycle import ProcessedCandidate
from contextual.models import CandidateRecord, CandidateStatus, RunResult, RunState, StopReason
from knowledge_items import ClusterConfig, KnowledgeBase, KnowledgeItemType
from utils import ModelConfig

from tests.contextual.helpers import label_book, make_facet, make_kb, make_settings


# ---- run identity ----

def _fp(config=None, kb_fp="kb1", prompt_fp="p1"):
    return cache.run_fingerprint(config or make_settings(), kb_fp, prompt_fp)


def test_fingerprint_follows_everything_that_changes_the_outcome(monkeypatch):
    base = _fp()
    assert _fp(make_settings(use_cache=True, max_concurrency=64)) == base   # non-semantic
    assert _fp(make_settings(refinement_threshold=0.5)) != base
    assert _fp(make_settings(seed=1)) != base
    assert _fp(kb_fp="kb2") != base and _fp(prompt_fp="p2") != base
    from optimization import irr
    monkeypatch.setattr(irr, "IRR_TEMPERATURE", 0.5)
    assert _fp() != base


def test_kb_fingerprint_is_content_hash():
    kb = make_kb()
    assert cache.kb_fingerprint(kb) == cache.kb_fingerprint(make_kb())
    changed = make_kb()
    changed.items[0].text = "different"
    assert cache.kb_fingerprint(changed) != cache.kb_fingerprint(kb)
    reordered = KnowledgeBase(items=list(reversed(make_kb().items)))
    assert cache.kb_fingerprint(reordered) == cache.kb_fingerprint(kb)


# ---- run folders ----

def _open(tmp_path, fingerprint="f1", fresh=False):
    return cache.RunFolder.open(tmp_path, "exp", "pipeline", fingerprint, {"config": {}},
                                fresh=fresh)


def _result(stop=StopReason.PROPOSAL_BUDGET):
    return RunResult.from_state(RunState(stop_reason=stop))


def test_interrupted_run_is_resumed_and_finished_run_is_not(tmp_path):
    first = _open(tmp_path)
    assert not first.resumed and first.info["status"] == "running"
    again = _open(tmp_path)
    assert again.resumed and again.path == first.path            # unfinished: resume
    assert _open(tmp_path, fresh=True).path != first.path         # --fresh: new folder
    latest = _open(tmp_path)
    assert latest.resumed                                          # the newest one
    latest.finish(_result())
    assert latest.info["status"] == "finished" and latest.info["stop_reason"] == "proposal_budget"
    assert cache.load_result(latest.path).stop_reason == StopReason.PROPOSAL_BUDGET
    assert not _open(tmp_path).resumed                             # finished: new folder
    assert not _open(tmp_path, fingerprint="f2").resumed           # other settings: new


def test_checkpoint_drops_a_half_written_last_line(tmp_path):
    run = _open(tmp_path)
    run.append({"n": 1})
    run.append({"n": 2})
    with open(run.path / "checkpoint.jsonl", "a") as f:
        f.write('{"n": 3, "pa')                                     # killed mid-write
    assert run.steps() == [{"n": 1}, {"n": 2}]
    run.append({"n": 3})
    assert run.steps() == [{"n": 1}, {"n": 2}, {"n": 3}]


def test_find_result_is_the_newest_finished_run_with_the_config(tmp_path):
    settings = make_settings()
    for n in range(2):
        run = cache.RunFolder.open(tmp_path, "exp", "pipeline", "f", fresh=True,
                                   info={"config": settings.semantic_dict()})
        run.finish(_result())
    newest = run.path
    cache.RunFolder.open(tmp_path, "exp", "pipeline", "f", fresh=True,
                         info={"config": settings.semantic_dict()})         # unfinished
    assert cache.find_result(tmp_path, "pipeline", settings) == newest
    assert cache.find_result(tmp_path, "agentic", settings) is None
    assert cache.find_result(tmp_path, "pipeline", make_settings(seed=3)) is None


def test_run_log_leaves_out_http_lines(tmp_path):
    run = _open(tmp_path)
    with run.logging():
        logging.getLogger("httpx").info("HTTP Request: POST ... 200 OK")
        logging.getLogger("contextual.test").info("attempt 1 accepted")
    text = run.log_path.read_text()
    assert "attempt 1 accepted" in text and "HTTP Request" not in text


# ---- checkpoint lines ----

def test_replaying_the_steps_rebuilds_the_state():
    cfg = make_settings(pool_size=2)
    evaluator = PoolEvaluator(config=cfg, focused_kb=make_kb())
    state, steps = RunState(fingerprint="f"), []
    before = cache.mark(state)
    # set up between steps (as the agent does before its first turn): saved with the next step
    state.runner_state["transcript"] = [{"role": "user", "text": "kickoff"}]
    for n, fid in enumerate(["A", "A2", "B"], 1):     # B replaces a pool member
        decision = evaluator.consider(state, ProcessedCandidate(
            facet=make_facet(fid), annotations=label_book([fid]), embedding=[float(n), 0.0]))
        state.embeddings[fid] = [float(n), 0.0]
        state.ledger.append(CandidateRecord(attempt_index=n, status=decision.status,
                                            facet_id=fid, title=fid))
        state.attempts = state.proposals = state.iteration = n
        state.runner_state["transcript"].append({"role": "tool", "content": fid})
        state.runner_state["turns"] = n
        state.usage = {"labeling": {"input_tokens": 10 * n}}
        steps.append(json.loads(json.dumps(cache.step_line(before, state))))
        before = cache.mark(state)
    assert [s.status for s in state.ledger][-1] in (CandidateStatus.REPLACED,
                                                    CandidateStatus.ACCEPTED,
                                                    CandidateStatus.SCHEMA_REJECTED)

    replayed = cache.replay(steps, "f")
    evaluator.initialize(replayed)
    assert replayed.to_dict() == state.to_dict()
    assert cache.replay(steps[:1], "f").pool_ids() == ["A"]


# ---- items and units ----

@dataclass
class FakeTask:
    id: str = "demo"
    name: str = "Demo Task"

    def get_dataset_path(self, dataset_dir) -> Path:
        return Path(dataset_dir) / f"{self.id}.json"


def _constructor(threshold=0.85) -> ConstructorConfig:
    return ConstructorConfig(
        use_vision=False,
        cluster_config=ClusterConfig(similarity_threshold=threshold, model_config=ModelConfig(
            category="embedding", provider="openai", model="fake-emb")),
        model_config=ModelConfig(category="vlm", provider="openai", model="fake-vlm"))


def test_items_and_units_are_reused_only_for_the_same_config(tmp_path, monkeypatch):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "demo.json").write_text("[]")
    monkeypatch.setattr(Config, "TASK_DATASETS_DIR", tmp_path / "data")
    monkeypatch.setattr(Config, "CACHE_DIR", tmp_path / "cache_2")
    extracted, clustered = [], []
    monkeypatch.setattr(construct_mod.ConstructKnowledgeBase, "extract", lambda self, task, it: (
        extracted.append(task.id) or list(make_kb().items)))
    monkeypatch.setattr(construct_mod.ConstructKnowledgeBase, "cluster", lambda self, items: (
        clustered.append(len(items)) or [replace(i, unit_id=f"u{n}") for n, i in enumerate(items)]))

    task, step = FakeTask(), KnowledgeItemType.STEP
    assert cache.knowledge_base(task, _constructor(), step, build=False) is None
    kb, sources = cache.knowledge_base(task, _constructor(), step)
    assert all(i.unit_id for i in kb) and set(sources) == {"items", "units"}
    root = tmp_path / "cache_2" / "demo"
    assert json.loads((root / sources["items"]).with_suffix(".json").read_text())["model"][
        "model"] == "fake-vlm"

    cache.knowledge_base(task, _constructor(), step)                # same config: reused
    assert extracted == ["demo"] and len(clustered) == 1
    cache.knowledge_base(task, _constructor(threshold=0.9), step)   # new clustering only
    assert extracted == ["demo"] and len(clustered) == 2
    (tmp_path / "data" / "demo.json").write_text("[1]")             # new dataset: new items
    cache.knowledge_base(task, _constructor(), step)
    assert extracted == ["demo", "demo"]
