"""
Experiment adapter for the prompt-based pipeline.

Loads experiment JSON files with a ``pipeline`` block, obtains the knowledge
base through ``contextual.kb``, and runs the pipeline with real (VLM) or
injected services.

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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from construct import ConstructorConfig

from knowledge_items import KnowledgeItemType, KnowledgeBase

from contextual import kb as kb_access
from contextual.kb import load_knowledge_base, lifecycle_services
from contextual.lifecycle import Embedder, VocabularyBuilder, Labeler
from contextual.models import RunResult

from .config import PipelineConfig
from .proposal import FacetProposer, VLMFacetProposer
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
# Knowledge base, results, services
# ---------------------------------------------------------------------------

RESULT_PREFIX = "pipeline"


def results_dir(spec: ExperimentSpec) -> Path:
    """Where the experiment's checkpoints and results live (no KB load)."""
    return kb_access.results_dir(spec.task_id, spec.constructor)


def cached_knowledge_base(spec: ExperimentSpec) -> Optional[KnowledgeBase]:
    """The experiment's knowledge base if already cached; never calls a model."""
    return kb_access.cached_knowledge_base(spec.task_id, spec.constructor, spec.item_type)


def find_result(spec: ExperimentSpec, directory: str | Path | None = None) -> Optional[Path]:
    """Newest saved pipeline result whose config matches ``spec``."""
    directory = directory if directory is not None else results_dir(spec)
    return kb_access.find_result(directory, RESULT_PREFIX, spec.pipeline)


@dataclass
class Services:
    proposer: FacetProposer
    vocab_builder: VocabularyBuilder
    labeler: Labeler
    embedder: Embedder


def vlm_services(config: PipelineConfig) -> Services:
    """Real model-backed services for a pipeline config."""
    lifecycle = lifecycle_services(config)
    return Services(
        proposer=VLMFacetProposer(config.proposal_model, config.max_concurrency),
        vocab_builder=lifecycle.vocab_builder,
        labeler=lifecycle.labeler,
        embedder=lifecycle.embedder,
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
) -> RunResult:
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
) -> RunResult:
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
