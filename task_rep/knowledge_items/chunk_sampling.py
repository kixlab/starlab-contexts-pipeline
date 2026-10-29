"""
Frame sampling strategies for different knowledge extraction granularities.

For ATOMIC and STEP: max_frames = frames per transcript segment
For TUTORIAL: max_frames = number of chunks across entire video
"""

from dataclasses import dataclass

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
