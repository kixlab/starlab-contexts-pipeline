"""
Sanity checker for video narration quality.

Filters out videos that lack sufficient meaningful narration, based on
their OpenAI Whisper transcript.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

DEFAULT_MIN_WORD_COUNT: int = 50
DEFAULT_MIN_WORDS_PER_MINUTE: float = 30.0


@dataclass
class NarrationCheckResult:
    passed: bool
    reason: Optional[str] = None  # set when passed=False
    stats: dict = field(default_factory=dict)


def check_narration_quality(
    transcript: list[dict],
    video_duration: float,
    min_word_count: int = DEFAULT_MIN_WORD_COUNT,
    min_words_per_minute: float = DEFAULT_MIN_WORDS_PER_MINUTE,
) -> NarrationCheckResult:
    """
    Check whether a video is sufficiently narrated.

    Uses two complementary signals from the OpenAI Whisper transcript:
      - Absolute word count  (catches very short stubs)
      - Words per minute     (catches long silences / music-only content)

    Args:
        transcript: List of segments with "text", "start", "end" keys.
        video_duration: Total video length in seconds.
        min_word_count: Minimum total words across all segments.
        min_words_per_minute: Minimum words per minute relative to video length.

    Returns:
        NarrationCheckResult with passed=True, or passed=False + reason.
    """
    stats: dict = {
        "word_count": 0,
        "words_per_minute": 0.0,
        "segment_count": 0,
        "video_duration": video_duration,
    }

    if not transcript:
        return NarrationCheckResult(
            passed=False,
            reason="No transcript segments found",
            stats=stats,
        )

    all_text = " ".join(seg["text"] for seg in transcript)
    word_count = len(all_text.split())
    words_per_minute = (word_count / video_duration * 60) if video_duration > 0 else 0.0

    stats = {
        "word_count": word_count,
        "words_per_minute": round(words_per_minute, 1),
        "segment_count": len(transcript),
        "video_duration": video_duration,
    }

    if word_count < min_word_count:
        return NarrationCheckResult(
            passed=False,
            reason=(
                f"Insufficient narration: {word_count} words "
                f"(minimum: {min_word_count})"
            ),
            stats=stats,
        )

    if words_per_minute < min_words_per_minute:
        return NarrationCheckResult(
            passed=False,
            reason=(
                f"Low narration density: {words_per_minute:.1f} words/min "
                f"(minimum: {min_words_per_minute:.1f})"
            ),
            stats=stats,
        )

    return NarrationCheckResult(passed=True, stats=stats)
