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
from typing import Dict, List

from utils import (
    ModelConfig,
    batch_run_llm_calls,
    random_uid,
)

from .prompts import (
    form_atomic_extraction_request,
    process_atomic_extraction_response,
    form_step_aggregation_request,
    process_step_aggregation_response,
    form_step_extraction_request_discovery,
    process_step_extraction_response_discovery,
    form_step_extraction_request_extraction,
    process_step_extraction_response_extraction,
    form_tutorial_summary_request,
    process_tutorial_summary_response,
)

from .schema import (
    KnowledgeItem,
    KnowledgeItemType,
    Video,
)
from .taxonomy import AtomicType, TYPE_DESCRIPTIONS

from .chunk_sampling import (
    ChunkConfig,
    FINE_CHUNK_CONFIG,
    COARSE_CHUNK_CONFIG,
)

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
    - STEP (discovery): Discover steps from videos (segment-based)
    - STEP (with taxonomy): Extract step details using canonical taxonomy (segment-based)
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
            "frames_per_group": self.chunk_config.frames_per_chunk,
            "target_chunks": self.chunk_config.size,
        }

    def _extract_atomic(self, task: str, videos: List[Video]) -> List[KnowledgeItem]:
        """Extract atomic information from videos."""
        batch_id = f"atomic_extraction_{random_uid()}"

        information_taxonomy = [
            {"name": type_name.value, "definition": desc}
            for type_name, desc in TYPE_DESCRIPTIONS.items()
        ]

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
        """Extract steps from videos with automatic three-phase discovery and extraction."""
        if not videos:
            return []

        # Phase 1: Discovery — identify raw steps per video
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

        # Build steps_by_video: video_id -> List[KnowledgeItem] (temporary discovery items)
        steps_by_video: Dict[str, List[KnowledgeItem]] = {
            video.id: result
            for video, result in zip(videos, batch_results)
        }

        # Phase 2: Aggregate into canonical taxonomy (corpus-level, single request)
        aggregate_batch_id = f"step_taxonomy_agg_{random_uid()}"
        request_args = [{
            "task": task,
            "videos": videos,
            "steps_by_video": steps_by_video,
        }]

        batch_results = batch_run_llm_calls(
            aggregate_batch_id,
            request_args,
            form_step_aggregation_request,
            process_step_aggregation_response,
            max_concurrency=self.max_concurrency,
            model_config=self.model_config,
        )

        step_taxonomy = batch_results[0]  # List[Dict[str, Any]]
        if not step_taxonomy:
            return []

        # Phase 3: Extraction using canonical taxonomy
        extraction_batch_id = f"step_extraction_{random_uid()}"
        request_args = [
            {
                "task": task,
                **self._base_video_args(video),
                "step_taxonomy": step_taxonomy,
            }
            for video in videos
        ]

        batch_results = batch_run_llm_calls(
            extraction_batch_id,
            request_args,
            form_step_extraction_request_extraction,
            process_step_extraction_response_extraction,
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
        - STEP: Three-phase (discovery → taxonomy aggregation → extraction)
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
