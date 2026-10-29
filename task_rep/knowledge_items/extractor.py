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
    MessageContent,
)

from prompts import (
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
    KnowledgeSource,
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

    def _extract_atomic(self, task: str, videos: List[Video]) -> List[KnowledgeItem]:
        """Extract atomic information from videos."""
        batch_id = f"atomic_extraction_{random_uid()}"

        request_args = []

        information_taxonomy = [
            {
                "name": type_name.value,
                "definition": desc,
            } for type_name, desc in TYPE_DESCRIPTIONS.items()
        ]

        for video in videos:
            video_content = video.format_to_message_content(
                use_vision=self.use_vision,
                frames_per_group=self.chunk_config.frames_per_chunk,
                target_chunks=self.chunk_config.size,
            )
            request_args.append({
                "task": task,
                "tutorial_content": video_content,
                "information_taxonomy": information_taxonomy,
            })

        batch_results = batch_run_llm_calls(
            batch_id,
            request_args,
            form_atomic_extraction_request,
            process_atomic_extraction_response,
            max_concurrency=self.max_concurrency,
            model_config=self.model_config,
        )
        
        items = []
        for video, result in zip(videos, batch_results):
            for idx in range(len(result)):
                atomic_type_str = result[idx].get("atomic_type")
                content = result[idx].get("content")
                start = result[idx].get("start")
                end = result[idx].get("end")

                if not atomic_type_str or not content:
                    continue
                
                # Convert to enum
                try:
                    atomic_type = AtomicType(atomic_type_str)
                except ValueError:
                    print(f"Invalid atomic type: {atomic_type_str}")
                    continue

                item_id = f"atomic:{atomic_type_str}-{idx}:{video.id}"

                items.append(KnowledgeItem(
                    id=item_id,
                    type=KnowledgeItemType.ATOMIC,
                    text=content,
                    source=KnowledgeSource(
                        video_id=video.id,
                        source_type="video",
                        local_index=idx,
                        time_range=(start, end),
                        parent_id=None,
                    ),
                    subtype=atomic_type,
                ))
        return items

    def _extract_steps(self, task: str, videos: List[Video]) -> List[KnowledgeItem]:
        """Extract steps from videos with automatic two-phase discovery and extraction."""

        if len(videos) == 0:
            return []

        # Phase 1: Discovery
        discovery_batch_id = f"step_discovery_{random_uid()}"
        request_args = []
        for video in videos:
            video_content = video.format_to_message_content(
                use_vision=self.use_vision,
                frames_per_group=self.chunk_config.frames_per_chunk,
                target_chunks=self.chunk_config.size,
            )
            request_args.append({
                "task": task,
                "tutorial_content": video_content,
            })

        batch_results = batch_run_llm_calls(
            discovery_batch_id,
            request_args,
            form_step_extraction_request_discovery,
            process_step_extraction_response_discovery,
            max_concurrency=self.max_concurrency,
            model_config=self.model_config,
        )

        steps_by_video = {}
        for video, result in zip(videos, batch_results):
            steps_by_video[video.id] = {
                "title": video.title,
                "steps": [],
            }
            for idx in range(len(result)):
                name = result[idx].get("name")
                content = result[idx].get("content")
                start = result[idx].get("start")
                end = result[idx].get("end")

                steps_by_video[video.id]["steps"].append({
                    "name": name,
                    "content": content,
                    "start": float(start),
                    "end": float(end),
                })

        # Phase 2: Aggregate taxonomy (corpus-level, single request)
        aggregate_batch_id = f"step_taxonomy_agg_{random_uid()}"

        # Format steps grouped by video
        steps_by_video_content = []
        
        for video_idx, video_data in enumerate(steps_by_video.values()):
            steps_by_video_content.append({"type": "text", "text": f"\nVideo {video_idx + 1}: {video_data['title']}"})
            for idx, step in enumerate(video_data["steps"]):
                steps_by_video_content.append({"type": "text", "text": f"{idx+1}. [{step.get('name')}] {step.get('content')}"})

        request_args = [{
            "task": task,
            "steps_by_tutorial": MessageContent(content=steps_by_video_content),
        }]

        batch_results = batch_run_llm_calls(
            aggregate_batch_id,
            request_args,
            form_step_aggregation_request,
            process_step_aggregation_response,
            max_concurrency=self.max_concurrency,
            model_config=self.model_config,
        )

        # batch_results[0] is the single corpus-level result
        step_taxonomy = []
        for item in batch_results[0]:
            name = item.get("name")
            definition = item.get("definition")
            step_taxonomy.append({
                "name": name,
                "definition": definition,
            })

        if not step_taxonomy:
            return []

        # Phase 3: Extraction
        extraction_batch_id = f"step_extraction_{random_uid()}"
        request_args = []
        for video in videos:
            video_content = video.format_to_message_content(
                use_vision=self.use_vision,
                frames_per_group=self.chunk_config.frames_per_chunk,
                target_chunks=self.chunk_config.size,
            )

            request_args.append({
                "task": task,
                "tutorial_content": video_content,
                "step_taxonomy": step_taxonomy,
            })

        batch_results = batch_run_llm_calls(
            extraction_batch_id,
            request_args,
            form_step_extraction_request_extraction,
            process_step_extraction_response_extraction,
            max_concurrency=self.max_concurrency,
            model_config=self.model_config,
        )

        items = []
        for video, result in zip(videos, batch_results):
            for idx in range(len(result)):
                name = result[idx].get("name")
                content = result[idx].get("content")
                start = result[idx].get("start")
                end = result[idx].get("end")
                item_id = f"step:{name}-{idx}:{video.id}"
                items.append(KnowledgeItem(
                    id=item_id,
                    type=KnowledgeItemType.STEP,
                    text=content,
                    source=KnowledgeSource(
                        video_id=video.id,
                        source_type="video",
                        local_index=idx,
                        time_range=(start, end),
                        parent_id=None,
                    ),
                    subtype=name,
                ))
        return items

    def _extract_tutorial(self, task: str, videos: List[Video]) -> List[KnowledgeItem]:
        """Extract tutorial summaries from videos."""
        batch_id = f"tutorial_extraction_{random_uid()}"

        request_args = []

        for video in videos:
            video_content = video.format_to_message_content(
                use_vision=self.use_vision,
                frames_per_group=self.chunk_config.frames_per_chunk,
                target_chunks=self.chunk_config.size,
            )
            request_args.append({
                "task": task,
                "tutorial_content": video_content,
            })

        batch_results = batch_run_llm_calls(
            batch_id,
            request_args, 
            form_tutorial_summary_request,
            process_tutorial_summary_response,
            max_concurrency=self.max_concurrency,
            model_config=self.model_config,
        )

        items = []
        for video, result in zip(videos, batch_results):
            assert len(result) == 1, "Expected exactly one result for tutorial summary"
            content = result[0].get("content")
            item_id = f"tutorial:{video.id}"
            items.append(KnowledgeItem(
                id=item_id,
                type=KnowledgeItemType.TUTORIAL,
                text=content,
                source=KnowledgeSource(
                    video_id=video.id,
                    source_type="video",
                    local_index=0,
                    time_range=(0, video.get_duration()),
                    parent_id=None,
                ),
                subtype=None,
            ))
        return items

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
