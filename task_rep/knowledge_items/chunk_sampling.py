"""
Frame sampling strategies for different knowledge extraction granularities.

For ATOMIC and STEP: max_frames = frames per transcript segment
For TUTORIAL: max_frames = number of chunks across entire video
"""

from dataclasses import dataclass
from enum import Enum
from typing import List
from .schema import VideoFrame

@dataclass
class ChunkConfig:
    """Configuration for chunks."""
    size: int = 0 # 0 means use all transcript segments (up to max)
    frames_per_chunk: int = 1

# Default configs for each granularity
ATOMIC_CHUNK_CONFIG = ChunkConfig(
    size=0,
    frames_per_chunk=1,
)
STEP_CHUNK_CONFIG = ChunkConfig(
    size=0,
    frames_per_chunk=1,
)
TUTORIAL_CHUNK_CONFIG = ChunkConfig(
    size=8,
    frames_per_chunk=1,
)


def sample_frames(
    frames: List[VideoFrame],
    start_time: float,
    end_time: float,
    max_frames: int,
) -> List[VideoFrame]:
    """
    Sample frames from a time range according to a chunk size.
    """
    # Filter frames in time range
    relevant_frames = [
        f for f in frames
        if start_time <= f.time <= end_time
    ]

    if not relevant_frames:
        return []

    return _sample_uniform(relevant_frames, max_frames)


def _sample_uniform(
    frames: List[VideoFrame],
    max_frames: int
) -> List[VideoFrame]:
    """Sample frames uniformly across the time range."""
    if len(frames) <= max_frames:
        return frames

    # Sample evenly
    step = len(frames) / max_frames
    indices = [int(i * step) for i in range(max_frames)]

    return [frames[i] for i in indices]
