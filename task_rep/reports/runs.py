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
from contextual import cache
from contextual.corpus import evaluated_segments, select_corpus
from contextual.kb import task_name
from contextual.models import RunResult, RunState
from contextual.settings import RunSettings
from knowledge_items import KnowledgeBase
from optimization import count_unique_units
from pipeline.fingerprint import prompt_fingerprint as pipeline_fingerprint

KINDS = {"pipeline": pexp, "agentic": aexp}


@dataclass
class LoadedRun:
    kind: str                  # "pipeline" | "agentic"
    settings: RunSettings      # the settings the run was made with
    result: RunResult
    path: Path                 # the run folder
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


def load_spec(kind: str, experiment: str):
    mod = KINDS[kind]
    spec = mod.load_experiment(mod.EXPERIMENTS_DIR / f"{experiment}.json")
    settings = spec.pipeline if kind == "pipeline" else spec.agentic
    return mod, spec, settings


def load_run(kind: str, experiment: str, run: Optional[str] = None) -> LoadedRun:
    """A finished run: ``run`` (a run folder name in the task's runs folder), or
    the newest finished run matching the experiment's settings."""
    mod, spec, settings = load_spec(kind, experiment)
    path = mod.results_dir(spec) / run if run else mod.find_result(spec)
    if path is None:
        raise FileNotFoundError(f"no finished run of {experiment} under its current settings")
    kb = mod.cached_knowledge_base(spec)
    if kb is None:
        raise FileNotFoundError(f"knowledge base for {experiment} is not cached")
    result = cache.load_result(path)
    # score with the settings the run was made with (they may differ from the file
    # now); the condition (context / feedback) never affects scoring
    settings = settings.__class__.from_dict({**settings.to_dict(), **{
        k: v for k, v in result.config.items()
        if k in settings.to_dict() and k not in ("context", "feedback")}})
    return LoadedRun(kind=kind, settings=settings, result=result, path=Path(path),
                     knowledge_base=kb, task=task_name(spec.task_id))


def save_run(kind: str, name: str, settings: RunSettings, corpus: KnowledgeBase,
             runs_dir: Path, state: RunState, focused: KnowledgeBase) -> Path:
    """Save ``state`` as a new finished run of ``name`` under ``settings``."""
    prompt_fp = (pipeline_fingerprint if kind == "pipeline" else agentic_fingerprint)(settings)
    kb_fp = cache.kb_fingerprint(corpus)
    fingerprint = cache.run_fingerprint(settings, kb_fp, prompt_fp)
    folder = cache.RunFolder.open(runs_dir, name, kind, fingerprint, fresh=True, info={
        "config": settings.semantic_dict(), "kb_fingerprint": kb_fp,
        "prompt_fingerprint": prompt_fp})
    state.fingerprint = fingerprint
    folder.finish(RunResult.from_state(state, kb_fingerprint=kb_fp, prompt_fingerprint=prompt_fp,
                                       config=settings.semantic_dict(),
                                       num_units=count_unique_units(focused)))
    return folder.path


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
