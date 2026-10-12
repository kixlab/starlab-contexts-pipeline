"""
Load finished runs of either runner, for reports.

``reports/`` consumes results of both runners, so unlike ``contextual/`` and
``analysis/`` it imports ``pipeline`` and ``agentic``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import agentic.experiments as aexp
import pipeline.experiments as pexp
from agentic.fingerprint import prompt_fingerprint as agentic_fingerprint
from contextual.corpus import evaluated_segments, select_corpus
from contextual.kb import task_name
from contextual.models import RunResult, RunState
from contextual.settings import RunSettings
from contextual.store import RunStore, kb_fingerprint
from knowledge_items import KnowledgeBase
from optimization import count_unique_units
from pipeline.fingerprint import prompt_fingerprint as pipeline_fingerprint

KINDS = {"pipeline": pexp, "agentic": aexp}


@dataclass
class LoadedRun:
    kind: str                  # "pipeline" | "agentic" (also the result-file prefix)
    settings: RunSettings      # the experiment file's settings
    result: RunResult
    path: Path                 # result file
    knowledge_base: KnowledgeBase   # full cached KB (before corpus selection)
    task: str = ""             # task name, as the prompts use it

    @property
    def corpus(self) -> KnowledgeBase:
        """The videos the run used (corpus sampling applied)."""
        return select_corpus(self.knowledge_base, self.settings)

    @property
    def focused(self) -> KnowledgeBase:
        """The segments the run was scored on (``evaluated_segments``)."""
        return evaluated_segments(self.corpus, self.settings)

    def state(self) -> Optional[RunState]:
        """The run's final checkpoint state, if it is still on disk."""
        ckpt = self.path.parent / f"{self.kind}_ckpt_{self.result.fingerprint}.json"
        if not ckpt.exists():
            return None
        return RunState.from_dict(json.loads(ckpt.read_text())["state"])


def load_spec(kind: str, experiment: str):
    mod = KINDS[kind]
    spec = mod.load_experiment(mod.EXPERIMENTS_DIR / f"{experiment}.json")
    settings = spec.pipeline if kind == "pipeline" else spec.agentic
    return mod, spec, settings


def load_run(kind: str, experiment: str, result_file: Optional[str] = None) -> LoadedRun:
    """A finished run: ``result_file`` (a file name in the experiment's results
    directory), or the newest result matching the experiment's settings."""
    mod, spec, settings = load_spec(kind, experiment)
    if result_file:
        path = mod.results_dir(spec) / result_file
    else:
        path = mod.find_result(spec)
        if path is None:
            raise FileNotFoundError(f"no result for {experiment} under its current settings")
    kb = mod.cached_knowledge_base(spec)
    if kb is None:
        raise FileNotFoundError(f"knowledge base for {experiment} is not cached")
    result = RunStore.load_result(path)
    # score with the settings the run was made with (they may differ from the file now);
    # the condition (context / feedback) never affects scoring, and older runs may
    # use condition values that are no longer valid, so it is read from result.config
    settings = settings.__class__.from_dict({**settings.to_dict(), **{
        k: v for k, v in result.config.items()
        if k in settings.to_dict() and k not in ("context", "feedback")}})
    return LoadedRun(kind=kind, settings=settings, result=result, path=Path(path),
                     knowledge_base=kb, task=task_name(spec.task_id))


def save_run(kind: str, settings: RunSettings, corpus: KnowledgeBase, results_dir: Path,
             state: RunState, focused: KnowledgeBase) -> Path:
    """Save ``state`` as a checkpoint and result under ``settings``."""
    fingerprint = (pipeline_fingerprint if kind == "pipeline" else agentic_fingerprint)(settings)
    store = RunStore(cache_dir=results_dir, config=settings, kb_fp=kb_fingerprint(corpus),
                     prompt_fp=fingerprint, prefix=kind)
    state.fingerprint = store.fingerprint
    store.save_checkpoint(state)
    result = RunResult.from_state(state, kb_fingerprint=store.kb_fp,
                                  prompt_fingerprint=fingerprint,
                                  config=settings.semantic_dict(),
                                  num_units=count_unique_units(focused))
    return store.save_result(result)


def parse_overrides(pairs) -> Dict[str, Any]:
    """``key=value`` strings to a dict, values parsed as JSON when possible."""
    out = {}
    for pair in pairs or []:
        key, _, raw = pair.partition("=")
        try:
            out[key] = json.loads(raw)
        except json.JSONDecodeError:
            out[key] = raw
    return out
