"""
Build datasets from tasks.json.

Simple: reads tasks.json, processes videos, creates datasets/{task_id}.json
Reuses existing data by default.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Literal

from .video_downloader import VideoDownloader, VideoDownloadConfig
from .task_manager import Task

from .whisper import transcribe_audio

def _make_youtube_url(video_id: str) -> str:
    """Construct YouTube URL from video ID."""
    return f"https://www.youtube.com/watch?v={video_id}"


def _fetch_youtube_transcript(video_id: str) -> dict:
    """
    Fetch original YouTube transcript using yt-dlp.

    Returns dict with 'transcript' and 'content' fields, or None if not available.
    """
    try:
        # Use yt-dlp to get subtitles
        cmd = [
            "yt-dlp",
            "--skip-download",
            "--write-auto-sub",
            "--sub-lang", "en",
            "--sub-format", "json3",
            "-o", f"static/videos/{video_id}/%(id)s",
            _make_youtube_url(video_id)
        ]

        result = subprocess.run(cmd, capture_output=True, timeout=60)

        # Try to load the subtitle file
        sub_file = Path(f"static/videos/{video_id}/{video_id}.en.json3")
        if sub_file.exists():
            with open(sub_file, "r", encoding="utf-8") as f:
                sub_data = json.load(f)

            # Parse transcript
            transcript = []
            content = ""
            for event in sub_data.get("events", []):
                if "segs" in event:
                    text = "".join(seg.get("utf8", "") for seg in event["segs"])
                    start = event.get("tStartMs", 0) / 1000.0
                    duration = event.get("dDurationMs", 0) / 1000.0

                    if text.strip():
                        transcript.append({
                            "text": text.strip(),
                            "start": start,
                            "end": start + duration
                        })
                        content += text + " "

            # Clean up subtitle file
            sub_file.unlink()

            if transcript:
                return {
                    "transcript": transcript,
                    "content": content.strip(),
                    "transcript_source": "youtube"
                }
    except Exception as e:
        print(f"    Could not fetch YouTube transcript: {e}")

    return None

def build_dataset(
    task: Task,
    base_dir: Path,
    video_download_config: VideoDownloadConfig,
    transcript_source: Literal["whisper", "youtube-or-whisper"] = "whisper",
    force_download: bool = False,
    force_extract: bool = False,
    force_transcripts: bool = False,
) -> str:
    """
    Build a complete dataset from a task definition.

    This function:
    1. Reads task from tasks.json
    2. Loads existing dataset if it exists (reuses by default)
    3. For each video:
       - Downloads video (if needed)
       - Extracts frames (if needed)
       - Fetches transcript (if needed)
    4. Saves complete dataset JSON

    Args:
        task_id: Task ID from tasks.json
        fps: Frames per second to extract
        transcript_source: How to get transcripts:
            - "whisper": Always use Whisper API (default, most reliable)
            - "youtube-or-whisper": Try YouTube transcript first, fall back to Whisper
        force_download: Re-download videos even if they exist
        force_extract: Re-extract frames even if they exist
        force_transcripts: Re-fetch transcripts even if they exist

    Returns:
        Path to the output dataset file

    Example:
        # Use Whisper (default)
        build_dataset("sushi_v1", fps=1.0)

        # Prefer YouTube transcripts
        build_dataset("sushi_v1", transcript_source="youtube-or-whisper")
    """

    output_path = base_dir / f"{task.id}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing dataset if it exists
    existing_dataset = {}
    if output_path.exists():
        print(f"Found existing dataset at {output_path}")
        with open(output_path, "r", encoding="utf-8") as f:
            dataset_list = json.load(f)
            existing_dataset = {v["id"]: v for v in dataset_list}
        print(f"Loaded {len(existing_dataset)} existing videos")

    # Initialize downloader
    downloader = VideoDownloader(video_download_config)

    # Process each video
    dataset = []
    skipped_videos = []

    for i, video_id in enumerate(task.video_ids):
        print(f"\n[{i+1}/{len(task.video_ids)}] Processing {video_id}")

        # Start with existing data or create new
        if video_id in existing_dataset:
            video_data = existing_dataset[video_id]
        else:
            video_data = {
                "id": video_id,
                "url": _make_youtube_url(video_id),
                "title": task.name,
            }

        # Check what we can reuse
        has_frames = video_data.get("frames") and not force_extract
        has_transcript = video_data.get("transcript") and not force_transcripts

        if has_frames and has_transcript:
            print(f"  Reusing all existing data for {video_id}")
            dataset.append(video_data)
            continue

        # Track if this video is completely unusable
        video_failed = False

        # Process video/frames if needed
        if not has_frames:
            try:
                skip_download = downloader.has_video(video_id) and not force_download
                skip_extract = downloader.has_frames(video_id) and not force_extract

                if skip_extract:
                    print(f"  Frames already exist, loading...")
                    frames = downloader.get_frames(video_id)
                elif skip_download:
                    print(f"  Video exists, extracting frames...")
                    frames = downloader.extract_frames(video_id)
                else:
                    print(f"  Downloading and extracting frames...")
                    frames = downloader.process_video(
                        video_id, force_download=force_download, force_extract=force_extract
                    )

                if not frames:
                    # Empty frames list means processing failed
                    print(f"  ✗ Failed to get frames")
                    video_failed = True
                else:
                    video_data["frames"] = frames
                    print(f"  ✓ {len(frames)} frames")

            except Exception as e:
                print(f"  ✗ Error processing video: {e}")
                video_failed = True

        # Skip this video entirely if it failed
        if video_failed:
            print(f"  Skipping video {video_id} (unavailable or failed to download)")
            skipped_videos.append(video_id)
            continue

        audio_path = downloader.get_audio(video_id)
        if not audio_path.exists():
            print(f"  ✗ Failed to get audio")
            video_failed = True
            skipped_videos.append(video_id)
            continue
        else:
            video_data["audio"] = audio_path
            print(f"  ✓ Audio: {audio_path}")

        # Fetch transcript if needed
        if not has_transcript:
            print(f"  Fetching transcript...")
            transcript_data = None

            # Try YouTube transcript first if requested
            if transcript_source == "youtube-or-whisper":
                print(f"    Trying YouTube transcript...")
                transcript_data = _fetch_youtube_transcript(video_id)
                if transcript_data and transcript_data.get("transcript"):
                    print(f"    ✓ YouTube transcript: {len(transcript_data['transcript'])} segments")

            # Fall back to Whisper if needed
            if not transcript_data or not transcript_data.get("transcript"):
                if transcript_source == "whisper" or transcript_source == "youtube-or-whisper":
                    if audio_path.exists():
                        transcript_data = transcribe_audio(audio_path)
                        if transcript_data.get("transcript"):
                            print(f"    ✓ Whisper transcript: {len(transcript_data['transcript'])} segments")

            # Update video data
            if transcript_data:
                video_data.update(transcript_data)
            else:
                video_data["transcript"] = []
                video_data["content"] = ""
                video_data["transcript_source"] = "none"
                print(f"    ! No transcript available")

        dataset.append(video_data)

    # Save dataset
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, indent=2, ensure_ascii=False, fp=f)

    print(f"\n✓ Dataset saved: {output_path}")
    print(f"  Total videos processed: {len(dataset)}/{len(task.video_ids)}")
    print(f"  With frames: {sum(1 for v in dataset if v.get('frames'))}")
    print(f"  With transcripts: {sum(1 for v in dataset if v.get('transcript'))}")
    print(f"  With audio: {sum(1 for v in dataset if v.get('audio'))}")

    # Show transcript sources
    sources = {}
    for v in dataset:
        src = v.get("transcript_source", "unknown")
        sources[src] = sources.get(src, 0) + 1
    if sources:
        print(f"  Transcript sources: {dict(sources)}")

    # Show skipped videos
    if skipped_videos:
        print(f"\n⚠ Skipped {len(skipped_videos)} video(s) (unavailable or failed to download):")
        for video_id in skipped_videos:
            print(f"    - {video_id}")

    return str(output_path)
