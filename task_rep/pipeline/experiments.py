"""
Outer experiment adapter.

The only layer that couples ``ConstructKnowledgeBase`` to the pipeline. Loads
experiment JSON files, obtains a knowledge base through the existing
constructor, runs the pipeline with real (VLM) or injected services, and
computes a concise summary.

Experiment file schema::

    {
      "name": "...", "task_id": "...", "description": "...", "tags": [...],
      "constructor": {...ConstructorConfig...},
      "pipeline": {...PipelineConfig minus embedding_model...}
    }

The dedup embedding model is not configurable separately: it is taken from
``constructor.cluster_config.model_config``. Legacy files with a ``discovery``
block are rejected (see ``LegacyExperimentError``).
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config import Config

from construct import ConstructKnowledgeBase, ConstructorConfig

from data_prep import TaskManager

from facets import VLMFacetLabeler, VLMFacetVocabularyBuilder

from knowledge_items import KnowledgeItemType, KnowledgeBase
from knowledge_items.io import get_clustering_cache_path, load_knowledge_items

from optimization import compute_tree_depth

from utils import ModelConfig
from utils.embeddings import get_embeddings

from .config import PipelineConfig
from .candidates import Embedder, VocabularyBuilder, Labeler
from .proposal import FacetProposer, VLMFacetProposer
from .models import PipelineResult
from .runner import PipelineRunner

EXPERIMENTS_DIR = Path(__file__).parent.parent / "experiments"


class LegacyExperimentError(ValueError):
    """Raised for experiment files still in the legacy ``discovery`` format."""


# ---------------------------------------------------------------------------
# Experiment spec
# ---------------------------------------------------------------------------

@dataclass
class ExperimentSpec:
    """One experiment file: a task, a knowledge-base constructor, a pipeline."""

    name: str
    task_id: str
    constructor: ConstructorConfig
    pipeline: PipelineConfig
    description: str = ""
    tags: List[str] = field(default_factory=list)

    @property
    def item_type(self) -> KnowledgeItemType:
        return self.pipeline.item_scope.item_type

    def to_dict(self) -> Dict[str, Any]:
        pipeline = self.pipeline.to_dict()
        pipeline.pop("embedding_model")
        return {
            "name": self.name,
            "task_id": self.task_id,
            "description": self.description,
            "tags": list(self.tags),
            "constructor": self.constructor.to_dict(),
            "pipeline": pipeline,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ExperimentSpec":
        if "discovery" in data:
            raise LegacyExperimentError(
                f"Experiment {data.get('name')!r} uses the legacy 'discovery' block; "
                "migrate it to a 'pipeline' block."
            )
        unknown = sorted(set(data) - {"name", "task_id", "description", "tags",
                                      "constructor", "pipeline"})
        if unknown:
            raise ValueError(f"Unknown experiment keys: {unknown}.")

        constructor = ConstructorConfig.from_dict(data["constructor"])
        pipeline_data = dict(data["pipeline"])
        if "embedding_model" in pipeline_data:
            raise ValueError(
                "pipeline.embedding_model is not configurable; dedup reuses "
                "constructor.cluster_config.model_config."
            )
        pipeline_data["embedding_model"] = constructor.cluster_config.model_config.to_dict()
        return ExperimentSpec(
            name=data["name"],
            task_id=data["task_id"],
            description=data.get("description", ""),
            tags=list(data.get("tags", [])),
            constructor=constructor,
            pipeline=PipelineConfig.from_dict(pipeline_data),
        )


def load_experiment(path: str | Path) -> ExperimentSpec:
    return ExperimentSpec.from_dict(json.loads(Path(path).read_text()))


def load_experiments_dir(
    directory: str | Path | None = None,
) -> Tuple[Dict[str, ExperimentSpec], List[Path]]:
    """Load every experiment file in ``directory``.

    Returns (specs by name, legacy-format files that were skipped). Duplicate
    names raise ``ValueError``.
    """
    directory = Path(directory) if directory is not None else EXPERIMENTS_DIR
    specs: Dict[str, ExperimentSpec] = {}
    legacy: List[Path] = []
    for path in sorted(directory.glob("*.json")):
        try:
            spec = load_experiment(path)
        except LegacyExperimentError:
            legacy.append(path)
            continue
        if spec.name in specs:
            raise ValueError(f"Duplicate experiment name {spec.name!r} in {path}.")
        specs[spec.name] = spec
    return specs, legacy


# ---------------------------------------------------------------------------
# Knowledge base + result locations
# ---------------------------------------------------------------------------

def _constructor(constructor_config: ConstructorConfig) -> ConstructKnowledgeBase:
    return ConstructKnowledgeBase(
        config=constructor_config,
        task_datasets_dir=Config.TASK_DATASETS_DIR,
        cache_dir=Config.CACHE_DIR,
    )


def load_knowledge_base(
    task_id: str,
    constructor_config: ConstructorConfig,
    item_type: KnowledgeItemType,
) -> tuple[Any, KnowledgeBase, Path]:
    """Load a task + its knowledge base via the existing constructor.

    May run extraction/clustering (VLM and embedding calls) when the
    constructor cache is missing.
    """
    constructor = _constructor(constructor_config)
    task = TaskManager(tasks_file=Config.TASKS_FILE).get_task(task_id)
    cache_dir = constructor.get_cache_dir(task)
    kb = constructor.get_knowledge_base(task, item_type)
    return task, kb, cache_dir


def results_dir(spec: ExperimentSpec) -> Path:
    """Where the experiment's checkpoints and results live (no KB load)."""
    task = TaskManager(tasks_file=Config.TASKS_FILE).get_task(spec.task_id)
    return _constructor(spec.constructor).get_cache_dir(task)


def cached_knowledge_base(spec: ExperimentSpec) -> Optional[KnowledgeBase]:
    """The experiment's knowledge base if already cached; never calls a model."""
    path = get_clustering_cache_path(
        cache_dir=results_dir(spec), item_type=spec.item_type,
        cluster_config=spec.constructor.cluster_config,
    )
    if not path.exists():
        return None
    return KnowledgeBase(items=load_knowledge_items(path=path))


def find_result(spec: ExperimentSpec, directory: str | Path | None = None) -> Optional[Path]:
    """Newest saved result in ``directory`` whose config matches ``spec``.

    Saved configs are normalized through ``PipelineConfig`` first, so a result
    written before a config field existed matches that field's default.
    Results for the same config under older prompts or knowledge bases also
    match; the newest one wins.
    """
    directory = Path(directory) if directory is not None else results_dir(spec)
    wanted = spec.pipeline.semantic_dict()
    matches = []
    for path in directory.glob("pipeline_result_*.json"):
        if _normalized_config(json.loads(path.read_text()).get("config", {})) == wanted:
            matches.append(path)
    return max(matches, key=lambda p: p.stat().st_mtime) if matches else None


def _normalized_config(saved: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """A saved semantic config with defaults filled in (None if unreadable)."""
    try:
        return PipelineConfig.from_dict(saved).semantic_dict()
    except (KeyError, TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Services
# ---------------------------------------------------------------------------

class ModelEmbedder:
    """Embedder backed by ``utils.embeddings.get_embeddings``."""

    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config

    def embed(self, texts: List[str]) -> np.ndarray:
        return get_embeddings(texts, model_config=self.model_config)


@dataclass
class Services:
    proposer: FacetProposer
    vocab_builder: VocabularyBuilder
    labeler: Labeler
    embedder: Embedder


def vlm_services(config: PipelineConfig) -> Services:
    """Real model-backed services for a pipeline config."""
    return Services(
        proposer=VLMFacetProposer(config.proposal_model, config.max_concurrency),
        vocab_builder=VLMFacetVocabularyBuilder(
            model_config=config.vocab_model, max_concurrency=config.max_concurrency,
        ),
        labeler=VLMFacetLabeler(
            model_config=config.labeling_model, max_concurrency=config.max_concurrency,
        ),
        embedder=ModelEmbedder(config.embedding_model),
    )


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def run_pipeline(
    config: PipelineConfig,
    task_name: str,
    knowledge_base: KnowledgeBase,
    cache_dir: str | Path,
    *,
    proposer: FacetProposer,
    vocab_builder: VocabularyBuilder,
    labeler: Labeler,
    embedder: Embedder,
    output_dir: str | Path | None = None,
) -> PipelineResult:
    """Invoke the contextual-unit runner with the given (real or fake) services.

    Checkpoints and results are written under ``output_dir`` (defaults to
    ``cache_dir``). Pass a scratch dir for throwaway/offline runs so they do not
    write into the shared knowledge-base cache.
    """
    runner = PipelineRunner(
        config=config,
        task=task_name,
        knowledge_base=knowledge_base,
        proposer=proposer,
        vocab_builder=vocab_builder,
        labeler=labeler,
        embedder=embedder,
        cache_dir=output_dir if output_dir is not None else cache_dir,
    )
    return runner.run()


def run_experiment(
    spec: ExperimentSpec,
    *,
    services: Optional[Services] = None,
    output_dir: str | Path | None = None,
) -> PipelineResult:
    """Load the experiment's knowledge base and run its pipeline.

    ``services`` defaults to the real VLM-backed services.
    """
    task, kb, cache_dir = load_knowledge_base(spec.task_id, spec.constructor, spec.item_type)
    services = services or vlm_services(spec.pipeline)
    return run_pipeline(
        spec.pipeline, task.name, kb, cache_dir,
        proposer=services.proposer, vocab_builder=services.vocab_builder,
        labeler=services.labeler, embedder=services.embedder,
        output_dir=output_dir,
    )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

@dataclass
class PipelineExperimentSummary:
    """Key metrics for quick comparison across runs."""

    num_units: int
    pool_facets: int
    iterations: int
    attempts: int
    proposals: int
    tree_nodes: int
    tree_depth: int
    stop_reason: str
    status_counts: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_units": self.num_units,
            "pool_facets": self.pool_facets,
            "iterations": self.iterations,
            "attempts": self.attempts,
            "proposals": self.proposals,
            "tree_nodes": self.tree_nodes,
            "tree_depth": self.tree_depth,
            "stop_reason": self.stop_reason,
            "status_counts": self.status_counts,
        }


def summarize(result: PipelineResult) -> PipelineExperimentSummary:
    counts = Counter(r.status.value for r in result.ledger)
    return PipelineExperimentSummary(
        num_units=result.num_units,
        pool_facets=len(result.pool),
        iterations=result.iterations,
        attempts=result.attempts,
        proposals=result.proposals,
        tree_nodes=len(result.tree.nodes) if result.tree else 0,
        tree_depth=compute_tree_depth(result.tree) if result.tree else 0,
        stop_reason=result.stop_reason.value if result.stop_reason else "none",
        status_counts=dict(counts),
    )
