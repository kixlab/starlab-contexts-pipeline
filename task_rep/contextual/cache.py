"""
Everything a run keeps on disk, under ``static/cache_2/<task>/``:

    items/<hash>.jsonl + <hash>.json     extracted items, and the config they were built with
    units/<hash>.jsonl + <hash>.json     the items with knowledge-unit ids (clustering)
    runs/<start time>__<experiment>/
        run.json          experiment, kind, fingerprint, config, the items/units files used
        checkpoint.jsonl  one line per completed step (append-only, flushed to disk)
        result.json       the finished run
        run.log           the run's log

Items and units are reused when their config matches exactly (the hash is of
the config). A run resumes only when it was interrupted: starting an
experiment replays the checkpoint of its newest run if that run has the same
fingerprint and no result yet; otherwise it starts a new folder. Model-call
logs stay in ``static/lm-batches``; each step names the ones it made.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from config import Config
from construct import ConstructKnowledgeBase, ConstructorConfig, extraction_prompt_fingerprint
from facets import AnnotationTable, Facet
from knowledge_items import KnowledgeBase, KnowledgeItemType, load_knowledge_items, save_knowledge_items
from optimization import irr
from utils.llm import SEED, logged_batches

from .models import CandidateRecord, RunResult, RunState, StopReason
from .settings import RunSettings

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "2"
COUNTERS = ("attempts", "proposals", "refusal_streak", "no_improve_streak", "iteration")


def _hash(obj: Any, n: int = 16) -> str:
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:n]


def task_dir(task_id: str) -> Path:
    return Path(Config.CACHE_DIR) / task_id


def runs_dir(task_id: str) -> Path:
    return task_dir(task_id) / "runs"


# ---------------------------------------------------------------------------
# Items and units
# ---------------------------------------------------------------------------

def kb_files(task, constructor_config: ConstructorConfig, item_type: KnowledgeItemType
             ) -> Dict[str, Tuple[str, Dict[str, Any]]]:
    """``{"items": (path, config), "units": (path, config)}``, paths relative to
    the task folder; each path is the hash of its config."""
    builder = ConstructKnowledgeBase(constructor_config, Config.TASK_DATASETS_DIR)
    items_config = {
        "task": task.id,
        "dataset": _hash(builder.dataset_path(task).read_text(encoding="utf-8")),
        "max_videos": constructor_config.max_videos,
        "item_type": item_type.value,
        "use_vision": constructor_config.use_vision,
        "model": constructor_config.model_config.to_dict(),
        "prompt": extraction_prompt_fingerprint(item_type, constructor_config.use_vision),
    }
    items_key = _hash(items_config)
    units_config = {"items": items_key, "cluster": constructor_config.cluster_config.to_dict()}
    return {"items": (f"items/{items_key}.jsonl", items_config),
            "units": (f"units/{_hash(units_config)}.jsonl", units_config)}


def knowledge_base(task, constructor_config: ConstructorConfig, item_type: KnowledgeItemType,
                   build: bool = True) -> Optional[Tuple[KnowledgeBase, Dict[str, str]]]:
    """The task's knowledge units and the files they came from (relative to the
    task folder). Builds missing items/units unless ``build`` is False."""
    files = kb_files(task, constructor_config, item_type)
    sources = {stage: path for stage, (path, _) in files.items()}
    root, reuse = task_dir(task.id), constructor_config.use_cache
    units_path = root / sources["units"]
    if reuse and units_path.exists():
        return KnowledgeBase(items=load_knowledge_items(units_path)), sources
    if not build:
        return None
    builder = ConstructKnowledgeBase(constructor_config, Config.TASK_DATASETS_DIR)
    items_path = root / sources["items"]
    if reuse and items_path.exists():
        items = load_knowledge_items(items_path)
    else:
        items = builder.extract(task, item_type)
        write_cached(items_path, items, files["items"][1])
    units = builder.cluster(items)
    write_cached(units_path, units, files["units"][1])
    return KnowledgeBase(items=units), sources


def write_cached(path: Path, items, config: Dict[str, Any], created: Optional[str] = None
                 ) -> None:
    """Items (JSONL) and, beside them, the config they were built with."""
    path.parent.mkdir(parents=True, exist_ok=True)
    save_knowledge_items(items, path)
    path.with_suffix(".json").write_text(json.dumps(
        {**config, "created": created or datetime.now().isoformat(timespec="seconds"),
         "count": len(items)}, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Run identity
# ---------------------------------------------------------------------------

def kb_fingerprint(kb: KnowledgeBase) -> str:
    """Content hash of a knowledge base (items ordered by id)."""
    items = sorted(kb.to_dict()["items"], key=lambda item: item["id"])
    return _hash(items)


def run_fingerprint(config: RunSettings, kb_fp: str, prompt_fp: str) -> str:
    """Everything that can change a run's outcome: knowledge base, config (minus
    settings such as ``use_cache``), models, rendered prompts, sampling."""
    return _hash({
        "schema_version": SCHEMA_VERSION,
        "kb": kb_fp,
        "prompt": prompt_fp,
        "config": config.semantic_dict(),
        "sampling": {"irr_temperature": irr.IRR_TEMPERATURE, "seed_base": SEED},
    }, n=24)


# ---------------------------------------------------------------------------
# Run folders
# ---------------------------------------------------------------------------

class RunFolder:
    """One run's folder: ``run.json``, ``checkpoint.jsonl``, ``result.json``, ``run.log``."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.resumed = False
        self.since: Optional[Dict[str, Any]] = None   # mark() after the last saved step

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def info(self) -> Dict[str, Any]:
        return json.loads((self.path / "run.json").read_text(encoding="utf-8"))

    @property
    def finished(self) -> bool:
        return (self.path / "result.json").exists()

    @property
    def log_path(self) -> Path:
        return self.path / "run.log"

    @staticmethod
    def open(directory: str | Path, name: str, kind: str, fingerprint: str,
             info: Dict[str, Any], fresh: bool = False) -> "RunFolder":
        """Resume the newest run of ``name`` if it was interrupted (same kind and
        fingerprint, no result); otherwise start a new folder."""
        directory = Path(directory)
        runs = sorted(directory.glob(f"*__{name}"), reverse=True) if directory.exists() else []
        for path in runs:
            run = RunFolder(path)
            if not (path / "run.json").exists():
                continue
            if run.info.get("kind") == kind and run.info.get("fingerprint") == fingerprint:
                if not fresh and not run.finished:
                    run.resumed = True
                    return run
                break                      # the newest matching run finished: start anew
        start = datetime.now()
        while True:     # one folder per second, so names sort in start order
            path = directory / f"{start.strftime('%Y-%m-%d_%H%M%S')}__{name}"
            try:
                path.mkdir(parents=True)
                break
            except FileExistsError:
                start += timedelta(seconds=1)
        run = RunFolder(path)
        run._write_json("run.json", {"experiment": name, "kind": kind, "fingerprint": fingerprint,
                                     "started": datetime.now().isoformat(timespec="seconds"),
                                     "status": "running", **info})
        return run

    def _write_json(self, filename: str, data: Dict[str, Any]) -> None:
        tmp = self.path / f".{filename}.tmp"
        tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.path / filename)

    # ---- checkpoint ----

    def steps(self) -> List[Dict[str, Any]]:
        """The completed steps; a half-written last line (an interrupted
        write) is dropped from the file."""
        path = self.path / "checkpoint.jsonl"
        if not path.exists():
            return []
        raw = path.read_bytes()
        lines, good = [], 0
        for line in raw.split(b"\n"):
            try:
                lines.append(json.loads(line))
            except ValueError:
                break
            good += len(line) + 1
        if good < len(raw):
            with open(path, "r+b") as f:
                f.truncate(good)
        return lines

    def append(self, step: Dict[str, Any]) -> None:
        with open(self.path / "checkpoint.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(step, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def finish(self, result: RunResult) -> Path:
        self._write_json("result.json", result.to_dict())
        self._write_json("run.json", {**self.info, "status": "finished",
                                      "finished": datetime.now().isoformat(timespec="seconds"),
                                      "stop_reason": result.stop_reason.value
                                      if result.stop_reason else None})
        return self.path / "result.json"

    # ---- log ----

    @contextlib.contextmanager
    def logging(self):
        """Append the block's log lines (without per-request HTTP lines) to run.log."""
        handler = logging.FileHandler(self.log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        handler.addFilter(lambda r: not (r.name.startswith("httpx") and r.levelno < logging.WARNING))
        root = logging.getLogger()
        level = root.level
        root.setLevel(min(level or logging.INFO, logging.INFO))
        root.addHandler(handler)
        try:
            yield self.log_path
        finally:
            root.removeHandler(handler)
            root.setLevel(level)
            handler.close()


def log_start(folder: RunFolder, kind: str, config: RunSettings, steps: int) -> None:
    info = folder.info
    sources = info.get("sources") or {}
    logger.info("Run %s: %s %s, %s", folder.path, kind, info["experiment"],
                f"resumed after {steps} step(s)" if folder.resumed else "fresh")
    if sources:
        logger.info("  knowledge base: %s", ", ".join(f"{k} {v}" for k, v in sources.items()))
    logger.info("  models: proposal %s, vocabulary %s, labeling %s; seed %s; budget %s; pool %s",
                config.proposal_model.model, config.vocab_model.model,
                config.labeling_model.model, config.seed, config.proposal_budget,
                config.pool_size)


def log_end(folder: RunFolder, result: RunResult) -> None:
    logger.info("Finished: %s; pool %d: %s", result.stop_reason.value if result.stop_reason
                else "?", len(result.pool), "; ".join(f.title for f in result.pool))
    logger.info("  result: %s", folder.path / "result.json")


def load_result(path: str | Path) -> RunResult:
    """A finished run's result (a run folder or its result.json)."""
    path = Path(path)
    if path.is_dir():
        path = path / "result.json"
    return RunResult.from_dict(json.loads(path.read_text(encoding="utf-8")))


def find_result(directory: str | Path, kind: str, settings: RunSettings) -> Optional[Path]:
    """The newest finished run folder of ``kind`` whose config equals ``settings``."""
    wanted = settings.semantic_dict()
    for path in sorted(Path(directory).glob("*__*"), reverse=True):   # newest first
        run = RunFolder(path)
        if run.finished and run.info.get("kind") == kind and run.info.get("config") == wanted:
            return path
    return None


# ---------------------------------------------------------------------------
# Checkpoint steps
# ---------------------------------------------------------------------------

def mark(state: RunState) -> Dict[str, Any]:
    """What a step line is measured against (taken before the step)."""
    return {
        "ledger": len(state.ledger),
        "pool": set(state.pool_ids()),
        "embeddings": set(state.embeddings),
        "usage": json.loads(json.dumps(state.usage)),
        "runner_state": {k: len(v) if isinstance(v, list) else json.dumps(v, sort_keys=True)
                         for k, v in state.runner_state.items()},
    }


def step_line(before: Dict[str, Any], state: RunState) -> Dict[str, Any]:
    """Everything the step changed: new ledger records, the pool order, the
    factors that entered it with their labels and embeddings, counters, usage,
    and runner-state changes (lists appended, other values replaced)."""
    added = [f for f in state.pool if f.id not in before["pool"]]
    added_ids = {f.id for f in added}
    labels: Dict[str, Dict[str, str]] = {fid: {} for fid in added_ids}
    for (item, fid), label in state.annotations.items():
        if fid in added_ids:
            labels[fid][item] = label
    runner = {}
    for key, value in state.runner_state.items():
        old = before["runner_state"].get(key)
        if isinstance(value, list) and isinstance(old, int):
            if len(value) > old:
                runner[key] = {"append": value[old:]}
        elif old != json.dumps(value, sort_keys=True):
            runner[key] = {"set": value}
    return {
        "ledger": [r.to_dict() for r in state.ledger[before["ledger"]:]],
        "pool": state.pool_ids(),
        "added": [f.to_dict() for f in added],
        "labels": labels,
        "embeddings": {k: v for k, v in state.embeddings.items()
                       if k not in before["embeddings"]},
        "counters": {c: getattr(state, c) for c in COUNTERS},
        "usage": state.usage,
        "stop_reason": state.stop_reason.value if state.stop_reason else None,
        "runner_state": runner,
    }


def replay(steps: List[Dict[str, Any]], fingerprint: str) -> RunState:
    """The run state after ``steps``; the caller recomputes the design space."""
    state = RunState(fingerprint=fingerprint)
    factors: Dict[str, Facet] = {}
    labels = AnnotationTable()
    for step in steps:
        state.ledger.extend(CandidateRecord.from_dict(r) for r in step["ledger"])
        for data in step["added"]:
            factors[data["id"]] = Facet.from_dict(data)
        for fid, rows in step["labels"].items():
            for item, label in rows.items():
                labels[(item, fid)] = label
        state.pool = [factors[fid] for fid in step["pool"]]
        state.embeddings.update(step["embeddings"])
        for counter, value in step["counters"].items():
            setattr(state, counter, value)
        state.usage = step["usage"]
        state.stop_reason = StopReason(step["stop_reason"]) if step["stop_reason"] else None
        for key, change in step["runner_state"].items():
            if "append" in change:
                state.runner_state.setdefault(key, []).extend(change["append"])
            else:
                state.runner_state[key] = change["set"]
    state.annotations = labels.filter_facets(state.pool_ids())
    return state


def start(folder: RunFolder, kind: str, config: RunSettings, fingerprint: str) -> RunState:
    """Log the run's header and rebuild its state from the saved steps (empty for
    a new run); the caller recomputes the design space."""
    steps = folder.steps()
    log_start(folder, kind, config, len(steps))
    state = replay(steps, fingerprint)
    folder.since = mark(state)
    return state


def run_step(folder: RunFolder, state: RunState, work: Callable[[RunState], Any],
             label: Callable[[RunState], str]) -> Any:
    """Run one step, then append what changed since the last saved step to the
    checkpoint, and the step's block to the log. The step's ``lm-batches`` ids
    go to ``runner_state["batches"]``."""
    before = folder.since if folder.since is not None else mark(state)
    with logged_batches() as batches:
        out = work(state)
    state.runner_state.setdefault("batches", []).append(batches)
    folder.append(step_line(before, state))
    folder.since = mark(state)
    logger.info(describe_step(before, state, batches, label(state)))
    return out


def describe_step(before: Dict[str, Any], state: RunState, batches: Dict[str, List[str]],
                  label: str) -> str:
    """The run log's block for one step: outcome, model-call logs, tokens."""
    lines = [f"── {label} ──"]
    for r in state.ledger[before["ledger"]:]:
        extra = f"; replaced {r.replaced_facet_id}" if r.replaced_facet_id else ""
        lines.append(f"  {r.status.value}: {r.title or '-'}{extra}"
                     + (f" ({r.note})" if r.note else ""))
    for stage, ids in batches.items():
        lines.append(f"  lm-batches {stage}: {', '.join(ids)}")
    spent = {k: 0 for k in ("input_tokens", "output_tokens", "thinking_tokens")}
    for stage, counts in state.usage.items():
        old = before["usage"].get(stage, {})
        for k in spent:
            spent[k] += counts.get(k, 0) - old.get(k, 0)
    lines.append("  tokens: " + ", ".join(f"{v:,} {k.split('_')[0]}" for k, v in spent.items()))
    return "\n".join(lines)
