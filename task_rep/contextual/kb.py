"""
Task and knowledge-base access, and the model-backed lifecycle services.

Experiment adapters call it with their own experiment specs; where things are
stored on disk is ``contextual.cache``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from config import Config

from construct import ConstructorConfig

from data_prep import TaskManager

from facets import VLMFacetLabeler, VLMFacetVocabularyBuilder

from knowledge_items import KnowledgeItemType, KnowledgeBase

from utils import ModelConfig
from utils.embeddings import get_embeddings

from . import cache
from .lifecycle import Embedder, Labeler, VocabularyBuilder
from .settings import RunSettings


# ---------------------------------------------------------------------------
# Tasks and knowledge bases
# ---------------------------------------------------------------------------

def get_task(task_id: str):
    return TaskManager(tasks_file=Config.TASKS_FILE).get_task(task_id)


def task_name(task_id: str) -> str:
    """The task's display name, as the prompts use it."""
    return get_task(task_id).name


def load_knowledge_base(
    task_id: str,
    constructor_config: ConstructorConfig,
    item_type: KnowledgeItemType,
) -> tuple[Any, KnowledgeBase, Dict[str, str]]:
    """The task, its knowledge base, and the items/units files it came from.

    Extracts and clusters (model calls) when the files are not cached yet.
    """
    task = get_task(task_id)
    kb, sources = cache.knowledge_base(task, constructor_config, item_type)
    return task, kb, sources


def cached_knowledge_base(
    task_id: str, constructor_config: ConstructorConfig, item_type: KnowledgeItemType,
) -> Optional[KnowledgeBase]:
    """The knowledge base if its units are cached; never calls a model."""
    found = cache.knowledge_base(get_task(task_id), constructor_config, item_type, build=False)
    return found[0] if found else None


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
