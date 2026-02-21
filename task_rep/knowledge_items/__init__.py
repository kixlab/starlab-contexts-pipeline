"""Knowledge extraction modules."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from .extractor import VLMKnowledgeExtractor  # noqa: F401

from .taxonomy import (  # noqa: F401
    AtomicType,
    AtomicCategory,
    TYPE_DESCRIPTIONS,
    TYPE_TO_CATEGORY,
    get_category,
    get_types_in_category,
)

from .schema import (  # noqa: F401
    KnowledgeBase,
    KnowledgeItem,
    KnowledgeItemType,
    KnowledgeItemSubtype,
    KnowledgeSource,
    Video,
    VideoFrame,
    VideoTranscriptSegment,
)

from .chunk_sampling import (  # noqa: F401
    ChunkConfig,
    ATOMIC_CHUNK_CONFIG,
    STEP_CHUNK_CONFIG,
    TUTORIAL_CHUNK_CONFIG,
)

from .clustering import (  # noqa: F401
    ClusterConfig,
    cluster_knowledge_items,
)

from .io import (  # noqa: F401
    save_knowledge_items,
    load_knowledge_items,
    load_dataset,
    get_extraction_cache_path,
    get_clustering_cache_path,
)


def extract_and_cluster_knowledge_items(
    task_id: str,
    granularity: KnowledgeItemType,
    videos: List[Video],
    base_dir: Path,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    cluster_config: Optional[ClusterConfig] = None,
    use_cache: bool = True,
    max_concurrency: int = 32,
) -> KnowledgeBase:
    """Extract and cluster knowledge items from a list of videos and return a KnowledgeBase."""

    knowledge_items = []

    # Get cache paths
    extraction_cache_path = get_extraction_cache_path(base_dir=base_dir, task_id=task_id, granularity=granularity)

    # Only get clustering cache path if cluster_config is provided
    clustering_cache_path = None
    if cluster_config is not None:
        clustering_cache_path = get_clustering_cache_path(base_dir=base_dir, task_id=task_id, granularity=granularity, config=cluster_config)

    ### Load clustered items from cache (if clustering is requested)
    if use_cache and clustering_cache_path is not None:
        try:
            knowledge_items = load_knowledge_items(input_path=clustering_cache_path)
            return KnowledgeBase(items=knowledge_items)
        except FileNotFoundError:
            pass

    ### Extract knowledge items
    if use_cache:
        try:
            knowledge_items = load_knowledge_items(input_path=extraction_cache_path)
        except FileNotFoundError:
            pass

    if len(knowledge_items) == 0:
        extractor = VLMKnowledgeExtractor(
            granularity=granularity,
            provider=provider,
            model=model,
            max_concurrency=max_concurrency,
        )
        knowledge_items = extractor.extract_batch(videos)

        save_knowledge_items(items=knowledge_items, output_path=extraction_cache_path)

    ### Cluster knowledge items (if config provided)
    if cluster_config is not None and len(knowledge_items) > 0:
        knowledge_items = cluster_knowledge_items(items=knowledge_items, config=cluster_config)

        if clustering_cache_path is not None:
            save_knowledge_items(items=knowledge_items, output_path=clustering_cache_path)

    return KnowledgeBase(items=knowledge_items)
