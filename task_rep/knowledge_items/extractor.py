"""
VLM-based knowledge extractor for multimodal tutorial videos.

This extractor uses vision-language models to extract knowledge at different
granularities while leveraging both transcript and visual information.

Design principles:
- Batch-first: Process entire corpus in parallel for efficiency
- Video-level: Extract all knowledge from a video in one pass
- Text-based: Store all information as natural text, not structured dicts
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from utils import (
    ModelConfig,
    batch_run_llm_calls,
    random_uid,
)

from .prompts import (
    form_atomic_extraction_request,
    process_atomic_extraction_response,
    form_step_extraction_request_discovery,
    process_step_extraction_response_discovery,
    form_tutorial_summary_request,
    process_tutorial_summary_response,
)

from .schema import (
    KnowledgeItem,
    KnowledgeItemType,
    Video,
)
from .taxonomy import TYPE_DESCRIPTIONS

from .chunk_sampling import ChunkConfig, FINE_CHUNK_CONFIG


def _information_taxonomy():
    return [{"name": atomic_type.value, "definition": info.definition, "examples": info.examples}
            for atomic_type, info in TYPE_DESCRIPTIONS.items()]


def extraction_request(item_type: KnowledgeItemType, task: str, video: Video,
                       use_vision: bool, chunk_config: ChunkConfig = FINE_CHUNK_CONFIG):
    """The request ``VLMKnowledgeExtractor`` sends for one video."""
    args = {"task": task, "video": video, "use_vision": use_vision, "chunk_config": chunk_config}
    if item_type == KnowledgeItemType.ATOMIC:
        return form_atomic_extraction_request(**args, information_taxonomy=_information_taxonomy())
    if item_type == KnowledgeItemType.STEP:
        return form_step_extraction_request_discovery(**args)
    return form_tutorial_summary_request(**args)


@dataclass
class VLMKnowledgeExtractor:
    """
    Batch-first VLM-based knowledge extractor.

    Design principles:
    - Batch processing: Extract from entire corpus in parallel
    - Video-level API calls: One API call per video
    - Segment-based content: ATOMIC/STEP use transcript segments, TUTORIAL uses uniform chunks
    - Text-based: Store all information as natural text

    Extraction modes:
    - ATOMIC: Extract atomic information with taxonomy classification (segment-based)
    - STEP: Discover steps from videos (segment-based, no aggregation)
    - TUTORIAL: Extract high-level tutorial summaries (chunk-based)
    """
    model_config: ModelConfig
    use_vision: bool = True
    max_concurrency: int = 32
    chunk_config: ChunkConfig = FINE_CHUNK_CONFIG

    def _base_video_args(self, video: Video) -> dict:
        """Common video formatting args shared across all extraction types."""
        return {
            "video": video,
            "use_vision": self.use_vision,
            "chunk_config": self.chunk_config,
        }

    def _extract_atomic(self, task: str, videos: List[Video]) -> List[KnowledgeItem]:
        """Extract atomic information from videos."""
        batch_id = f"atomic_extraction_{random_uid()}"

        information_taxonomy = _information_taxonomy()

        request_args = [
            {
                "task": task,
                **self._base_video_args(video),
                "information_taxonomy": information_taxonomy,
            }
            for video in videos
        ]

        batch_results = batch_run_llm_calls(
            batch_id,
            request_args,
            form_atomic_extraction_request,
            process_atomic_extraction_response,
            max_concurrency=self.max_concurrency,
            model_config=self.model_config,
        )

        return [item for result in batch_results for item in result]

    def _extract_steps(self, task: str, videos: List[Video]) -> List[KnowledgeItem]:
        """Extract steps from videos (single-phase discovery, no aggregation)."""
        if not videos:
            return []

        # Single phase: discover steps per video (subtype=None, like tutorials)
        discovery_batch_id = f"step_discovery_{random_uid()}"
        request_args = [
            {"task": task, **self._base_video_args(video)}
            for video in videos
        ]

        batch_results = batch_run_llm_calls(
            discovery_batch_id,
            request_args,
            form_step_extraction_request_discovery,
            process_step_extraction_response_discovery,
            max_concurrency=self.max_concurrency,
            model_config=self.model_config,
        )

        return [item for result in batch_results for item in result]

    def _extract_tutorial(self, task: str, videos: List[Video]) -> List[KnowledgeItem]:
        """Extract tutorial summaries from videos."""
        batch_id = f"tutorial_extraction_{random_uid()}"

        request_args = [
            {"task": task, **self._base_video_args(video)}
            for video in videos
        ]

        batch_results = batch_run_llm_calls(
            batch_id,
            request_args,
            form_tutorial_summary_request,
            process_tutorial_summary_response,
            max_concurrency=self.max_concurrency,
            model_config=self.model_config,
        )

        return [item for result in batch_results for item in result]

    def extract_batch(self, task: str, videos: List[Video], item_type: KnowledgeItemType) -> List[KnowledgeItem]:
        """
        Extract knowledge from multiple videos in parallel (batch processing).

        Automatically handles the appropriate extraction workflow for each item_type:
        - ATOMIC: Single-phase extraction
        - STEP: Single-phase discovery (no aggregation, subtype=None)
        - TUTORIAL: Single-phase extraction

        This is the main method for efficient extraction.
        """
        if not videos:
            return []

        if item_type == KnowledgeItemType.ATOMIC:
            return self._extract_atomic(task=task, videos=videos)
        elif item_type == KnowledgeItemType.STEP:
            return self._extract_steps(task=task, videos=videos)
        elif item_type == KnowledgeItemType.TUTORIAL:
            return self._extract_tutorial(task=task, videos=videos)
