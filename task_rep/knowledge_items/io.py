"""
I/O utilities for knowledge items.

Saving and loading knowledge items (JSONL) and task datasets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from .schema import KnowledgeItem, Video, VideoTranscriptSegment, VideoFrame


def save_knowledge_items(
    items: List[KnowledgeItem],
    path: str | Path,
) -> None:
    """
    Save knowledge items to JSONL format.

    Each line is a JSON object representing one knowledge item.

    Args:
        items: List of knowledge items to save
        path: Path to output JSONL file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")


def load_knowledge_items(
    path: str | Path,
) -> List[KnowledgeItem]:
    """
    Load knowledge items from JSONL format.

    Args:
        path: Path to input JSONL file

    Returns:
        List of knowledge items
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Knowledge items file not found: {path}")

    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                items.append(KnowledgeItem.from_dict(data))

    return items


def load_task_dataset(path: str | Path) -> List[Video]:
    """
    Load videos from dataset JSON file.

    Args:
        path: Path to dataset JSON file

    Returns:
        List of Video objects with transcript segments and frames
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
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