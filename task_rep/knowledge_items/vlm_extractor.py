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

import asyncio
from dataclasses import dataclass
from task_rep.knowledge_items.schema import VideoTranscriptSegment
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from uuid import uuid4

from .schema import (
    KnowledgeItem,
    KnowledgeItemType,
    KnowledgeSource,
    Video,
    VideoTranscriptSegment,
)
from .taxonomy import AtomicType, TYPE_DESCRIPTIONS
from .chunk_sampling import (
    ChunkConfig,
    sample_frames,
    ATOMIC_CHUNK_CONFIG,
    STEP_CHUNK_CONFIG,
    TUTORIAL_CHUNK_CONFIG,
)
from task_rep.utils.llm import (
    MessageContent,
    llm_enqueue_request,
    process_api_requests,
    llm_get_responses,
)
from task_rep.prompts import atomic_extraction, step_aggregation, step_extraction, tutorial_summary


# ============= Pydantic Models for Structured Extraction =============

class AtomicInformation(BaseModel):
    """Single atomic piece of information with taxonomy classification."""
    atomic_type: str = Field(description="Classification from taxonomy (e.g., 'Method - Instruction')")
    content: str = Field(description="Natural text content of this atomic information")
    start: float = Field(description="Start time of the atomic information in the video")
    end: float = Field(description="End time of the atomic information in the video")


class AtomicExtractionResult(BaseModel):
    """Result of extracting all atomic information from a video."""
    items: List[AtomicInformation] = Field(description="All atomic information items found in the video")


class StepInformation(BaseModel):
    """Information about a single step (used for both discovery and detailed extraction)."""
    name: str = Field(description="Name of this step (e.g., 'Prepare rice')")
    content: str = Field(description="Natural text content describing this step")
    start: float = Field(description="Start time of the step in the video")
    end: float = Field(description="End time of the step in the video")


class StepExtractionResult(BaseModel):
    """Result of step extraction from a video."""
    steps: List[StepInformation] = Field(description="All steps found in the video")


class StepTaxonomy(BaseModel):
    """Aggregated taxonomy of canonical step names across all videos."""
    steps: List[str] = Field(description="List of canonical step names")


class TutorialInformation(BaseModel):
    """High-level tutorial summary."""
    content: str = Field(description="Natural text summary of the entire tutorial")


# ============= VLM Extractor Implementation =============

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

    granularity: KnowledgeItemType = KnowledgeItemType.ATOMIC
    use_vision: bool = True
    provider: Optional[str] = None
    model: Optional[str] = None
    max_concurrency: int = 32

    # Chunk sampling config
    chunk_config: Optional[ChunkConfig] = None

    def __post_init__(self):
        """Initialize chunk sampling configs and default model based on provider."""
        # Set default model if not provided
        if self.provider is None:
            self.provider = "openai"
        
        if self.model is None:
            if self.provider == "openai":
                self.model = "gpt-4o-mini"
            elif self.provider == "anthropic":
                self.model = "us.anthropic.claude-3-5-haiku-20241022-v1:0"

        # Set chunk config based on granularity
        if self.chunk_config is None:
            if self.granularity == KnowledgeItemType.ATOMIC:
                self.chunk_config = ATOMIC_CHUNK_CONFIG
            elif self.granularity == KnowledgeItemType.STEP:
                self.chunk_config = STEP_CHUNK_CONFIG
            elif self.granularity == KnowledgeItemType.TUTORIAL:
                self.chunk_config = TUTORIAL_CHUNK_CONFIG

    def _create_interleaved_content(self, video: Video, instruction: str) -> MessageContent:
        """
        Create interleaved content with chunks and transcript temporally aligned.

        Unified approach for all granularities:
        - Groups transcript segments (1 segment per group for ATOMIC/STEP, merged for TUTORIAL)
        - Samples frames within each group
        - Creates temporal markers based on actual segment boundaries

        For ATOMIC/STEP: group_size=1, max_frames per segment
        For TUTORIAL: groups merged to hit target chunk count, 1 chunk per group

        Returns MessageContent with interleaved structure for better temporal grounding.
        """
        if self.use_vision and not video.frames:
            raise ValueError("When vision is enabled, video must have frames")

        if not video.transcript:
            raise ValueError("Video must have transcript")

        frames_per_group = self.chunk_config.frames_per_chunk
        target_chunks = self.chunk_config.size
        segment_groups = []
        
        # Determine how to group segments
        if target_chunks is None:
            # i.e., ATOMIC/STEP or TUTORIAL with number of chunks larger than number of segments
            target_chunks = len(video.transcript)
            # Use each segment as its own chunk
            segment_groups = [[seg] for seg in video.transcript]
        else:
            # i.e., TUTORIAL
            segment_groups = self._group_segments(video.transcript, target_chunks)

        # Build truly interleaved content
        content_blocks = [{"type": "text", "text": instruction + "\n\n"}]

        for group in segment_groups:
            # Get time range for this group
            group_start = group[0].start
            group_end = group[-1].end

            # Combine transcript text from all segments in group
            group_text = " ".join(seg.text for seg in group)

            # Sample frames within this group's time range
            group_frames = sample_frames(
                frames=video.frames,
                start_time=group_start,
                end_time=group_end,
                max_frames=self.chunk_config.frames_per_chunk
            )

            if group_frames or group_text:
                # Add temporal marker
                content_blocks.append({
                    "type": "text",
                    "text": f"\n--- Time {self._format_time(group_start)}-{self._format_time(group_end)} ---\n"
                })

                if group_text:
                    # Add group transcript
                    content_blocks.append({"type": "text", "text": f"{group_text}\n"})

                if group_frames:
                    # Interleave frames with time markers
                    for frame in group_frames:
                        content_blocks.append({
                            "type": "text",
                            "text": f"[Visual frame at {self._format_time(frame.time)}]\n"
                        })
                        content_blocks.append({"type": "image_path", "path": frame.path})

        return MessageContent(content=content_blocks)

    def _group_segments(self, segments: List[VideoTranscriptSegment], target_chunks: int) -> List[List[VideoTranscriptSegment]]:
        """
        Group segments to achieve target number of chunks.

        Merges adjacent segments to create approximately target_chunks groups,
        maintaining natural segment boundaries.
        """
        if len(segments) <= target_chunks:
            # Already have fewer segments than target - use each segment as its own group
            return [[seg] for seg in segments]

        # Calculate how many segments per group on average to achieve target_chunks
        segments_per_group = len(segments) / target_chunks

        groups = []
        current_group = []
        current_count = 0

        for i, segment in enumerate[VideoTranscriptSegment](segments):
            current_group.append(segment)
            current_count += 1

            # Check if we should start a new group
            # Either we've accumulated enough segments, or we're at the last segment
            is_last_segment = (i == len(segments) - 1)
            has_enough_segments = current_count >= segments_per_group
            remaining_segments = len(segments) - i - 1
            remaining_groups = target_chunks - len(groups) - 1

            # Start new group if we have enough segments and there are enough remaining
            # to fill the rest of the groups
            if (has_enough_segments and remaining_segments >= remaining_groups) or is_last_segment:
                groups.append(current_group)
                current_group = []
                current_count = 0

                # Stop if we've reached target chunks
                if len(groups) >= target_chunks:
                    break

        return groups

    def _format_time(self, seconds: float) -> str:
        """Format seconds as MM:SS."""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}:{secs:02d}"

    def _enqueue_atomic_request(self, batch_id: str, video: Video) -> str:
        """Enqueue request to extract atomic information from a video."""
        # Build taxonomy reference
        taxonomy_desc = "\n".join([
            f"- {type_name.value}: {desc.split('Example:')[0].strip()}"
            for type_name, desc in TYPE_DESCRIPTIONS.items()
        ])
        instruction = atomic_extraction.get_instruction(taxonomy_desc)
        content = self._create_interleaved_content(video, instruction)

        llm_enqueue_request(
            batch_id=batch_id,
            messages=[
                {"role": "system", "content": atomic_extraction.get_system_message()},
                {"role": "user", "content": content}
            ],
            response_format=AtomicExtractionResult,
            model=self.model,
            options={"video_id": video.id}
        )

        return video.id

    def _enqueue_step_discovery_request(self, batch_id: str, video: Video) -> str:
        """Enqueue request to discover steps from a video."""
        instruction = step_extraction.get_instruction(phase="discovery")
        content = self._create_interleaved_content(video, instruction)

        llm_enqueue_request(
            batch_id=batch_id,
            messages=[
                {"role": "system", "content": step_extraction.get_system_message(phase="discovery")},
                {"role": "user", "content": content}
            ],
            response_format=StepExtractionResult,
            model=self.model,
            options={"video_id": video.id, "phase": "discovery"}
        )

        return video.id

    def _enqueue_step_extraction_request(self, batch_id: str, video: Video, step_taxonomy: List[str]) -> str:
        """Enqueue request to extract step details from a video using canonical taxonomy."""
        instruction = step_extraction.get_instruction(phase="extraction", step_taxonomy=step_taxonomy)
        content = self._create_interleaved_content(video, instruction)

        llm_enqueue_request(
            batch_id=batch_id,
            messages=[
                {"role": "system", "content": step_extraction.get_system_message(phase="extraction")},
                {"role": "user", "content": content}
            ],
            response_format=StepExtractionResult,
            model=self.model,
            options={"video_id": video.id, "phase": "extraction"}
        )

        return video.id

    def _enqueue_tutorial_request(self, batch_id: str, video: Video) -> str:
        """Enqueue request to extract tutorial summary from a video."""
        instruction = tutorial_summary.get_instruction()
        content = self._create_interleaved_content(video, instruction)

        llm_enqueue_request(
            batch_id=batch_id,
            messages=[
                {"role": "system", "content": tutorial_summary.get_system_message()},
                {"role": "user", "content": content}
            ],
            response_format=TutorialInformation,
            model=self.model,
            options={"video_id": video.id}
        )

        return video.id

    def _process_atomic_responses(
        self,
        responses: Dict[str, Dict],
        video_map: Dict[str, Video]
    ) -> List[KnowledgeItem]:
        """Process responses from atomic extraction batch."""
        items = []

        for request_id, response_data in responses.items():
            video_id = response_data.get("options", {}).get("video_id")
            if not video_id or video_id not in video_map:
                continue

            video = video_map[video_id]
            result = response_data.get("response", {})

            # Process each atomic item
            for idx, atomic_info in enumerate(result.get("items", [])):
                atomic_type_str = atomic_info.get("atomic_type")
                content = atomic_info.get("content")

                if not atomic_type_str or not content:
                    continue
                
                # Convert to enum
                try:
                    atomic_type = AtomicType(atomic_type_str)
                except ValueError:
                    print(f"Invalid atomic type: {atomic_type_str}")
                    continue

                item_id = f"{video.id}:atomic:{idx}"
                items.append(KnowledgeItem(
                    id=item_id,
                    type=KnowledgeItemType.ATOMIC,
                    text=content,  # Store as natural text
                    source=KnowledgeSource(video_id=video.id, local_index=idx),
                    subtype=atomic_type,
                ))

        return items

    def _extract_discovered_steps(
        self,
        responses: Dict[str, Dict],
        video_map: Dict[str, Video]
    ) -> List[Dict]:
        """Extract raw discovered steps from discovery responses with video provenance."""
        all_steps = []

        for response_data in responses.values():
            video_id = response_data.get("options", {}).get("video_id")
            if not video_id or video_id not in video_map:
                continue

            video = video_map[video_id]
            result = response_data.get("response", {})
            steps_list = result.get("steps", [])

            for step in steps_list:
                all_steps.append({
                    "name": step.get("name"),
                    "content": step.get("content"),
                    "video_id": video_id,
                    "video_title": video.title
                })

        return all_steps

    def _process_step_extraction_responses(
        self,
        responses: Dict[str, Dict],
        video_map: Dict[str, Video]
    ) -> List[KnowledgeItem]:
        """Process responses from step extraction (not discovery)."""
        items = []

        for request_id, response_data in responses.items():
            video_id = response_data.get("options", {}).get("video_id")

            if not video_id or video_id not in video_map:
                continue

            video = video_map[video_id]
            result = response_data.get("response", {})

            # Create items for each extracted step
            for idx, step_info in enumerate(result.get("steps", [])):
                step_name = step_info.get("name")
                content = step_info.get("content")

                if not step_name or not content:
                    continue

                item_id = f"{video.id}:step:{idx}"
                items.append(KnowledgeItem(
                    id=item_id,
                    type=KnowledgeItemType.STEP,
                    text=content,  # Store as natural text
                    source=KnowledgeSource(video_id=video.id, local_index=idx),
                    subtype=step_name,  # Canonical step name
                ))

        return items

    def _process_tutorial_responses(
        self,
        responses: Dict[str, Dict],
        video_map: Dict[str, Video]
    ) -> List[KnowledgeItem]:
        """Process responses from tutorial extraction batch."""
        items = []

        for request_id, response_data in responses.items():
            video_id = response_data.get("options", {}).get("video_id")
            if not video_id or video_id not in video_map:
                continue

            video = video_map[video_id]
            result = response_data.get("response", {})
            content = result.get("content")

            if content:
                item_id = f"{video.id}:summary"
                items.append(KnowledgeItem(
                    id=item_id,
                    type=KnowledgeItemType.TUTORIAL,
                    text=content,  # Store as natural text
                    source=KnowledgeSource(video_id=video.id),
                    subtype=None,
                ))

        return items

    async def _extract_atomic(self, videos: List[Video]) -> List[KnowledgeItem]:
        """Extract atomic information from videos."""
        batch_id = f"atomic_extraction_{uuid4().hex[:8]}"
        video_map = {v.id: v for v in videos}

        for video in videos:
            self._enqueue_atomic_request(batch_id, video)

        await process_api_requests(batch_id, self.max_concurrency, self.provider)
        responses = llm_get_responses(batch_id)
        return self._process_atomic_responses(responses, video_map)

    async def _extract_steps(self, videos: List[Video]) -> List[KnowledgeItem]:
        """Extract steps from videos with automatic two-phase discovery and extraction."""
        video_map = {v.id: v for v in videos}

        # Phase 1: Discovery
        discovery_batch_id = f"step_discovery_{uuid4().hex[:8]}"
        for video in videos:
            self._enqueue_step_discovery_request(discovery_batch_id, video)

        await process_api_requests(discovery_batch_id, self.max_concurrency, self.provider)
        discovery_responses = llm_get_responses(discovery_batch_id)
        discovered_steps = self._extract_discovered_steps(discovery_responses, video_map)

        if not discovered_steps:
            return []

        # Phase 2: Aggregate taxonomy
        step_taxonomy = await self._aggregate_step_taxonomy(discovered_steps)
        if not step_taxonomy:
            return []

        # Phase 3: Extraction
        extraction_batch_id = f"step_extraction_{uuid4().hex[:8]}"
        for video in videos:
            self._enqueue_step_extraction_request(extraction_batch_id, video, step_taxonomy)

        await process_api_requests(extraction_batch_id, self.max_concurrency, self.provider)
        extraction_responses = llm_get_responses(extraction_batch_id)
        return self._process_step_extraction_responses(extraction_responses, video_map)

    async def _extract_tutorial(self, videos: List[Video]) -> List[KnowledgeItem]:
        """Extract tutorial summaries from videos."""
        batch_id = f"tutorial_extraction_{uuid4().hex[:8]}"
        video_map = {v.id: v for v in videos}

        for video in videos:
            self._enqueue_tutorial_request(batch_id, video)

        await process_api_requests(batch_id, self.max_concurrency, self.provider)
        responses = llm_get_responses(batch_id)
        return self._process_tutorial_responses(responses, video_map)

    async def extract_batch(self, videos: List[Video]) -> List[KnowledgeItem]:
        """
        Extract knowledge from multiple videos in parallel (batch processing).

        Automatically handles the appropriate extraction workflow for each granularity:
        - ATOMIC: Single-phase extraction
        - STEP: Two-phase (discovery → taxonomy → extraction)
        - TUTORIAL: Single-phase extraction

        This is the main method for efficient extraction.
        """
        if not videos:
            return []

        if self.granularity == KnowledgeItemType.ATOMIC:
            return await self._extract_atomic(videos)
        elif self.granularity == KnowledgeItemType.STEP:
            return await self._extract_steps(videos)
        elif self.granularity == KnowledgeItemType.TUTORIAL:
            return await self._extract_tutorial(videos)

        return []

    def extract_from_video(self, video: Video) -> List[KnowledgeItem]:
        """
        Extract knowledge from a single video (for backward compatibility).

        For efficient batch processing, use extract_batch() instead.
        """
        # Use batch extraction with a single video
        import asyncio
        return asyncio.run(self.extract_batch([video]))


    async def _aggregate_step_taxonomy(
        self,
        discovered_steps: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Aggregate discovered steps from multiple videos into a canonical taxonomy.

        Args:
            discovered_steps: List of raw step dicts with 'name', 'content', 'video_id', 'video_title'

        Returns:
            List of canonical step names
        """
        if not discovered_steps:
            return []

        # Get instruction with formatted steps
        instruction = step_aggregation.get_instruction(discovered_steps)

        # Create batch for single aggregation request
        batch_id = f"step_taxonomy_agg_{uuid4().hex[:8]}"

        llm_enqueue_request(
            batch_id=batch_id,
            messages=[
                {"role": "system", "content": step_aggregation.get_system_message()},
                {"role": "user", "content": MessageContent(content=[{"type": "text", "text": instruction}])}
            ],
            response_format=StepTaxonomy,
            model=self.model,
        )

        # Process request
        await process_api_requests(batch_id=batch_id, provider=self.provider)

        # Get response
        responses = llm_get_responses(batch_id)

        # Extract step names
        for response_data in responses.values():
            result = response_data.get("response", {})
            return result.get("steps", [])

        return []
