"""
I/O utilities for knowledge items.

Handles saving and loading knowledge items with caching support.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from .schema import KnowledgeItem, KnowledgeItemType, Video, VideoTranscriptSegment, VideoFrame
from .clustering import ClusterConfig


def save_knowledge_items(
    items: List[KnowledgeItem],
    output_path: str | Path,
) -> None:
    """
    Save knowledge items to JSONL format.

    Each line is a JSON object representing one knowledge item.

    Args:
        items: List of knowledge items to save
        output_path: Path to output JSONL file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")


def load_knowledge_items(
    input_path: str | Path,
) -> List[KnowledgeItem]:
    """
    Load knowledge items from JSONL format.

    Args:
        input_path: Path to input JSONL file

    Returns:
        List of knowledge items
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Knowledge items file not found: {input_path}")

    items = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                items.append(KnowledgeItem.from_dict(data))

    return items


def get_extraction_cache_path(
    base_dir: str | Path,
    task_id: str,
    granularity: KnowledgeItemType,
) -> Path:
    """
    Generate cache path for extracted knowledge items.

    Args:
        base_dir: Base directory for cache files
        task_id: Task identifier
        granularity: Knowledge item type (ATOMIC, STEP, TUTORIAL)

    Returns:
        Path to cache file

    Example:
        >>> get_extraction_cache_path("cache", "sushi_v1", KnowledgeItemType.ATOMIC)
        Path('cache/sushi_v1/extracted_atomic.jsonl')
    """
    base_dir = Path(base_dir)
    task_dir = base_dir / task_id
    return task_dir / f"extracted_{granularity.value}.jsonl"


def get_clustering_cache_path(
    base_dir: str | Path,
    task_id: str,
    granularity: KnowledgeItemType,
    config: ClusterConfig,
) -> Path:
    """
    Generate cache path for clustered knowledge items.

    Includes clustering config in filename for different thresholds.

    Args:
        base_dir: Base directory for cache files
        task_id: Task identifier
        granularity: Knowledge item type
        config: Clustering configuration

    Returns:
        Path to cache file

    Example:
        >>> config = ClusterConfig(similarity_threshold=0.85)
        >>> get_clustering_cache_path("cache", "sushi_v1", KnowledgeItemType.ATOMIC, config)
        Path('cache/sushi_v1/clustered_atomic_sim85_textembedding3small.jsonl')
    """
    base_dir = Path(base_dir)
    task_dir = base_dir / task_id
    suffix = config.cache_suffix()
    return task_dir / f"clustered_{granularity.value}_{suffix}.jsonl"


def load_dataset(dataset_path: str | Path) -> List[Video]:
    """
    Load videos from dataset JSON file.

    Args:
        dataset_path: Path to dataset JSON file (typically from data_prep)

    Returns:
        List of Video objects with transcript segments and frames

    Example:
        >>> videos = load_dataset("static/datasets/sushi_roll_3.json")
        >>> len(videos)
        5
    """
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    videos = []
    for entry in raw:
        metadata = entry.get("metadata", {})
        transcript_segments = [
            VideoTranscriptSegment(
                text=seg["text"],
                start=float(seg["start"]),
                end=float(seg["end"]),
            )
            for seg in entry.get("transcript", [])
        ]

        frames = []
        if "frames" in entry:
            frames = [
                VideoFrame(
                    path=f["path"],
                    time=float(f["time"]),
                    caption=f.get("caption"),
                )
                for f in entry["frames"]
            ]

        video = Video(
            id=entry["id"],
            url=entry["url"],
            title=entry["title"],
            metadata=metadata,
            transcript=transcript_segments,
            frames=frames,
        )
        videos.append(video)

    return videos