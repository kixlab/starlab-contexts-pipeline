"""
Knowledge-base access, saved-result lookup, and model-backed lifecycle services.

The only module that couples ``ConstructKnowledgeBase`` to the contextual-unit
runners; experiment adapters call it with their own experiment specs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import numpy as np

from config import Config

from construct import ConstructKnowledgeBase, ConstructorConfig

from data_prep import TaskManager

from facets import VLMFacetLabeler, VLMFacetVocabularyBuilder

from knowledge_items import KnowledgeItemType, KnowledgeBase
from knowledge_items.io import get_clustering_cache_path, load_knowledge_items

from utils import ModelConfig
from utils.embeddings import get_embeddings

from .lifecycle import Embedder, Labeler, VocabularyBuilder
from .settings import RunSettings


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


def task_name(task_id: str) -> str:
    """The task's display name, as the prompts use it."""
    return TaskManager(tasks_file=Config.TASKS_FILE).get_task(task_id).name


def results_dir(task_id: str, constructor_config: ConstructorConfig) -> Path:
    """Where a task's checkpoints and results live (no KB load)."""
    task = TaskManager(tasks_file=Config.TASKS_FILE).get_task(task_id)
    return _constructor(constructor_config).get_cache_dir(task)


def cached_knowledge_base(
    task_id: str, constructor_config: ConstructorConfig, item_type: KnowledgeItemType,
) -> Optional[KnowledgeBase]:
    """The knowledge base if already cached; never calls a model."""
    path = get_clustering_cache_path(
        cache_dir=results_dir(task_id, constructor_config), item_type=item_type,
        cluster_config=constructor_config.cluster_config,
    )
    if not path.exists():
        return None
    return KnowledgeBase(items=load_knowledge_items(path=path))


def find_result(directory: str | Path, prefix: str, settings: RunSettings) -> Optional[Path]:
    """Newest ``{prefix}_result_*.json`` in ``directory`` whose config matches.

    Saved configs are normalized through ``type(settings)`` first, so a result
    written before a config field existed matches that field's default.
    Results for the same config under older prompts or knowledge bases also
    match; the newest one wins.
    """
    wanted = settings.semantic_dict()
    matches = []
    for path in Path(directory).glob(f"{prefix}_result_*.json"):
        saved = json.loads(path.read_text()).get("config", {})
        if _normalized_config(type(settings), saved) == wanted:
            matches.append(path)
    return max(matches, key=lambda p: p.stat().st_mtime) if matches else None


def _normalized_config(
    cls: Type[RunSettings], saved: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """A saved semantic config with defaults filled in (None if unreadable)."""
    try:
        return cls.from_dict(saved).semantic_dict()
    except (KeyError, TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Lifecycle services
# ---------------------------------------------------------------------------

class ModelEmbedder:
    """Embedder backed by ``utils.embeddings.get_embeddings``."""

    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config

    def embed(self, texts: List[str]) -> np.ndarray:
        return get_embeddings(texts, model_config=self.model_config)


@dataclass
class LifecycleServices:
    vocab_builder: VocabularyBuilder
    labeler: Labeler
    embedder: Embedder


def lifecycle_services(settings: RunSettings) -> LifecycleServices:
    """Real model-backed vocabulary builder, labeler, and embedder."""
    return LifecycleServices(
        vocab_builder=VLMFacetVocabularyBuilder(
            model_config=settings.vocab_model, max_concurrency=settings.max_concurrency,
        ),
        labeler=VLMFacetLabeler(
            model_config=settings.labeling_model, max_concurrency=settings.max_concurrency,
        ),
        embedder=ModelEmbedder(settings.embedding_model),
    )
