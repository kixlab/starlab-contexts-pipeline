from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

from task_rep.knowledge_items import save_knowledge_items
from task_rep.knowledge_items import TutorialVideoExtractor, construct_knowledge_base
from task_rep.knowledge_items import KnowledgeBase, Video, VideoTranscriptSegment, VideoFrame


@dataclass
class ExtractionConfig:
    """
    Minimal configuration for running knowledge item extraction.

    - input_path: JSON file containing a list of video dicts with keys
      (id, url, title, original_title, content, transcript)
    - output_path: where to write the JSONL knowledge item dataset
    - make_atomic_items: whether to derive atomic items from tutorial summaries
    """

    input_path: str
    output_path: str
    make_atomic_items: bool = True


def _load_video_corpus(path: str | Path) -> List[Video]:
    """
    Load a list of Video objects from a JSON file.

    Supports both old format (without frames) and new format (with frames).
    """
    in_path = Path(path)
    with in_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    videos: List[Video] = []
    for entry in raw:
        # Load transcript segments
        transcript_segments = [
            VideoTranscriptSegment(
                text=seg["text"],
                start=float(seg["start"]),
                end=float(seg["end"]),
            )
            for seg in entry.get("transcript", [])
        ]

        # Load frames if available
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
            original_title=entry.get("original_title", entry["title"]),
            content=entry.get("content", "") or "",
            transcript=transcript_segments,
            frames=frames,
        )
        videos.append(video)

    return videos


def run_extraction(config: ExtractionConfig) -> KnowledgeBase:
    """
    High-level entrypoint to build a knowledge base from a video corpus file.
    """
    videos = _load_video_corpus(config.input_path)
    extractor = TutorialVideoExtractor(
        make_summary_items=True,
        make_atomic_items=config.make_atomic_items,
    )
    knowledge_base = construct_knowledge_base(videos, extractor)

    save_knowledge_items(knowledge_base.items, config.output_path)
    return knowledge_base

