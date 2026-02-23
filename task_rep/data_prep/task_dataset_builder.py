"""
Build datasets from tasks.json.

Simple: reads tasks.json, processes videos, creates datasets/{task_id}.json
Reuses existing data by default.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Literal

from .video_downloader import (
    VideoDownloader,
)
from .task_manager import (
    Task,
)

def build_task_dataset(
    task: Task,
    downloader: VideoDownloader,
    dataset_dir: str | Path,
    fps: float = 1.0,
    transcript_source: Literal["openai", "youtube"] = "openai",
    force_download: bool = False,
) -> Path:
    """
    Build a complete dataset from a task definition.

    This function:
    1. Reads task from tasks.json
    2. For each video:
        - Downloads metadata, video, audio and YouTube transcript (if needed or force_download is True)
        - Extracts frames according to fps
        - Extracts the transcript according to transcript_source
    3. Saves complete dataset JSON

    Args:
        task: Task object from tasks.json
        downloader: VideoDownloader object
        fps: Frames per second to extract
        transcript_source: Which transcripts to use: "openai" or "youtube"
        force_download: Re-download videos even if they exist
    """

    output_path: Path = task.get_dataset_path(dataset_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Process each video
    dataset = []
    skipped_videos = []

    for i, video_id in enumerate(task.video_ids):
        print(f"\n[{i+1}/{len(task.video_ids)}] Processing {video_id}")

        # Rate-limit: sleep between downloads to avoid YouTube bot detection
        if i > 0:
            delay = random.uniform(3, 7)
            print(f"  Sleeping {delay:.1f}s to avoid rate limiting...")
            time.sleep(delay)

        video_data = {
            "id": video_id,
            "url": downloader.get_video_url(video_id),
        }

        try:
            downloader.download_video(video_id, force=force_download)


            metadata = downloader.get_video_metadata_summary(video_id)
            if metadata is None:
                raise Exception(f"No metadata found for {video_id}")

            video_data["metadata"] = metadata
            print(f"  ✓ Video metadata summary: {metadata['title']} ({metadata['duration']}s)")
            video_data["title"] = metadata["title"]

            frames = downloader.extract_frames(video_id, fps=fps)
            video_data["frames"] = frames
            print(f"  ✓ {len(frames)} frames")

            transcript_data = downloader.get_transcript(video_id, transcript_source)

            if transcript_data is None:
                raise Exception(f"No transcript found for {video_id}")

            transcript = transcript_data["transcript"]

            video_data["transcript"] = transcript
            video_data["transcript_source"] = transcript_source

            print(f"  ✓ {transcript_source} transcript: {len(transcript)} segments")
        except Exception as e:
            print(f"  ✗ Error processing video: {e}")
            skipped_videos.append(video_id)
            continue

        dataset.append(video_data)

    # Save dataset
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, indent=2, ensure_ascii=False, fp=f)

    print(f"\n✓ Dataset saved: {output_path}")
    print(f"  Total videos processed: {len(dataset)}/{len(task.video_ids)}")

    # Show skipped videos
    if skipped_videos:
        print(f"\n⚠ Skipped {len(skipped_videos)} video(s) (unavailable or failed to download):")

    return Path(output_path)
