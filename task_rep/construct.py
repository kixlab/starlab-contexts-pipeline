"""
Knowledge-base construction: extract items from a task's videos, then cluster
them into knowledge units. Where the results are cached is
``contextual.cache.knowledge_base``'s business.
"""

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
    KnowledgeItem,
    KnowledgeItemType,
    ClusterConfig,
    Video,
    load_task_dataset,
    cluster_knowledge_items,
)
from knowledge_items.extractor import extraction_request
from knowledge_items.schema import VideoFrame, VideoTranscriptSegment
from utils.llm import MessageContent

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
    """False rebuilds the items and units even if they are cached."""
    max_concurrency: int = _CC_MAX_CONCURRENCY

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


def extraction_prompt_fingerprint(item_type: KnowledgeItemType, use_vision: bool) -> str:
    """Hash of the extraction request rendered on a fixed two-segment video, so
    the items cache follows the prompt (not comments or refactors)."""
    import hashlib, json
    video = Video(id="fixture", url="", title="Fixture video",
                  transcript=[VideoTranscriptSegment(text="First step.", start=0.0, end=5.0),
                              VideoTranscriptSegment(text="Second step.", start=5.0, end=10.0)],
                  frames=[VideoFrame(path="fixture.jpg", time=2.0)])
    request = extraction_request(item_type, task="fixture task", video=video,
                                 use_vision=use_vision)
    messages = [{"role": m["role"], "content": m["content"].content
                 if isinstance(m["content"], MessageContent) else m["content"]}
                for m in request["messages"]]
    blob = json.dumps({"messages": messages,
                       "schema": request["response_format"].model_json_schema()},
                      sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


class ConstructKnowledgeBase:
    """Builds a task's knowledge items (extraction) and units (clustering)."""

    def __init__(self, config: ConstructorConfig, task_datasets_dir: str | Path):
        self.config = config
        self.task_datasets_dir = Path(task_datasets_dir)
        self._extractor = VLMKnowledgeExtractor(
            use_vision=self.config.use_vision,
            max_concurrency=self.config.max_concurrency,
            model_config=self.config.model_config,
        )

    def dataset_path(self, task: Task) -> Path:
        return task.get_dataset_path(self.task_datasets_dir)

    def extract(self, task: Task, item_type: KnowledgeItemType) -> List[KnowledgeItem]:
        """Extract items from the task's videos (calls the model)."""
        videos: List[Video] = load_task_dataset(path=self.dataset_path(task))
        if self.config.max_videos is not None:
            videos = videos[: self.config.max_videos]
        logger.info(f"Extracting knowledge items from {len(videos)} videos")
        return self._extractor.extract_batch(task=task.name, videos=videos, item_type=item_type)

    def cluster(self, items: List[KnowledgeItem]) -> List[KnowledgeItem]:
        """The items with knowledge-unit ids (calls the embedding model)."""
        if not items:
            return []
        return cluster_knowledge_items(items=items, config=self.config.cluster_config)
