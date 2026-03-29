from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from pathlib import Path

from utils import ModelConfig

from data_prep import (
    Task,
)

from knowledge_items import (
    VLMKnowledgeExtractor,
    KnowledgeItemType,
    ClusterConfig,
    KnowledgeBase,
    Video,
    get_extraction_cache_path,
    get_clustering_cache_path,
    load_knowledge_items,
    load_task_dataset,
    save_knowledge_items,
    cluster_knowledge_items,
)

logger = logging.getLogger(__name__)

# ---- ConstructorConfig defaults (single source of truth) ----
_CC_MAX_VIDEOS = None
_CC_USE_CACHE = True
_CC_MAX_CONCURRENCY = 32

@dataclass
class ConstructorConfig:
    """Knowledge item construction settings."""

    use_vision: bool
    cluster_config: ClusterConfig
    model_config: ModelConfig

    max_videos: Optional[int] = _CC_MAX_VIDEOS
    """Limit to first N videos. None = all videos in the file."""

    use_cache: bool = _CC_USE_CACHE
    max_concurrency: int = _CC_MAX_CONCURRENCY

    def cache_suffix(self) -> str:
        max_videos = str(self.max_videos) if self.max_videos is not None else ""
        use_vision = "vision" if self.use_vision else "no_vision"
        model_suffix = self.model_config.cache_suffix()
        return f"{max_videos}_{use_vision}_{model_suffix}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "use_vision": self.use_vision,
            "use_cache": self.use_cache,
            "max_videos": self.max_videos,
            "max_concurrency": self.max_concurrency,
            "cluster_config": self.cluster_config.to_dict(),
            "model_config": self.model_config.to_dict(),
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ConstructorConfig":
        return ConstructorConfig(
            use_vision=data["use_vision"],
            cluster_config=ClusterConfig.from_dict(data["cluster_config"]),
            model_config=ModelConfig.from_dict(data["model_config"]),
            max_videos=data.get("max_videos", _CC_MAX_VIDEOS),
            use_cache=data.get("use_cache", _CC_USE_CACHE),
            max_concurrency=data.get("max_concurrency", _CC_MAX_CONCURRENCY),
        )


class ConstructKnowledgeBase:
    def __init__(self,
        config: ConstructorConfig,
        task_datasets_dir: str | Path,
        cache_dir: str | Path
    ):
        self.config = config
        self.cache_dir = Path(cache_dir)
        self.task_datasets_dir = Path(task_datasets_dir)
        self._extractor = VLMKnowledgeExtractor(
            use_vision=self.config.use_vision,
            max_concurrency=self.config.max_concurrency,
            model_config=self.config.model_config,
        )

    def get_cache_dir(self, task: Task) -> Path:
        suffix = self.config.cache_suffix()
        return task.get_cache_dir(self.cache_dir, suffix)

    def get_knowledge_base(self, task: Task, item_type: KnowledgeItemType) -> KnowledgeBase:
        use_cache = self.config.use_cache
        cluster_config = self.config.cluster_config
        cache_dir = self.get_cache_dir(task)
        task_dataset_path = task.get_dataset_path(self.task_datasets_dir)
        knowledge_items = []

        # Get cache paths
        extraction_cache_path: Path = get_extraction_cache_path(
            cache_dir=cache_dir, item_type=item_type
        )   
        clustering_cache_path: Path = get_clustering_cache_path(
            cache_dir=cache_dir, item_type=item_type, cluster_config=cluster_config
        )

        if use_cache and clustering_cache_path.exists():
            knowledge_items = load_knowledge_items(path=clustering_cache_path)
            return KnowledgeBase(items=knowledge_items)

        if use_cache and extraction_cache_path.exists():
            knowledge_items = load_knowledge_items(path=extraction_cache_path)
        else:
            max_videos = self.config.max_videos
            videos: List[Video] = load_task_dataset(path=task_dataset_path)
            if max_videos is not None:
                videos = videos[: max_videos]

            logger.info(f"Constructing knowledge base for {len(videos)} videos")

            knowledge_items = self._extractor.extract_batch(
                task=task.name, videos=videos, item_type=item_type
            )
            save_knowledge_items(items=knowledge_items, path=extraction_cache_path)

        if len(knowledge_items) == 0:
            return KnowledgeBase(items=[])

        ### Cluster knowledge items
        knowledge_items = cluster_knowledge_items(items=knowledge_items, config=cluster_config)
        save_knowledge_items(items=knowledge_items, path=clustering_cache_path)

        return KnowledgeBase(items=knowledge_items)