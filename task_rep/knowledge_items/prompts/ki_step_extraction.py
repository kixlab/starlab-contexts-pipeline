"""
Prompts for step-level extraction from tutorial videos.

This includes both step discovery (identifying what steps exist) and
step extraction based on a step taxonomy.
"""

import logging
from typing import Any, Dict, List

from utils import MessageContent

from .formatters import taxonomy_to_message_content, video_to_message_content
from .ki_pydantic import StepExtractionResult

from knowledge_items.schema import KnowledgeItem, KnowledgeItemType, KnowledgeSource, Video
from knowledge_items.chunk_sampling import ChunkConfig

logger = logging.getLogger(__name__)


SYSTEM_MESSAGE_DISCOVERY = """You are an expert at analyzing tutorial videos and identifying procedural steps about the task: {task}. You excel at recognizing the high-level structure of tutorials, breaking down complex procedures into clear steps, and understanding both what is shown visually and what is explained in narration. Inputs are structured with XML tags — use the tag name and id attribute to identify each object and its boundaries."""

USER_MESSAGE_DISCOVERY = """You are given temporally aligned frames and transcript from a tutorial video. Analyze the video and identify ALL procedural steps relevant to completing the task.

For each step:
1. Provide a short, clear step name
2. Write a comprehensive, but concise description of what this step involves
3. Provide the start and end times of the step in the video

Use both visual cues from frames and transcript information to identify steps.
Make sure that steps have a clear start and end time in the video, and each segment of the video assigned to exactly one step.
Return ALL the task-relevant steps found in the video with detailed descriptions and start/end times.

INPUT:
- Tutorial:
{tutorial}"""


def form_step_extraction_request_discovery(
    task: str,
    video: Video,
    use_vision: bool,
    chunk_config: ChunkConfig,
    **kwargs,
) -> Dict[str, Any]:
    """Form request to discover steps from a tutorial."""
    tutorial_content = video_to_message_content(video, use_vision, chunk_config)
    messages = [
        {
            "role": "system",
            "content": MessageContent.from_template(SYSTEM_MESSAGE_DISCOVERY, task=task),
        },
        {
            "role": "user",
            "content": MessageContent.from_template(
                USER_MESSAGE_DISCOVERY,
                tutorial=tutorial_content,
            ),
        },
    ]
    return {
        "messages": messages,
        "response_format": StepExtractionResult,
    }


def process_step_extraction_response_discovery(
    response: Dict[str, Any],
    video: Video,
    **kwargs,
) -> List[KnowledgeItem]:
    """Process step discovery response, returning temporary KnowledgeItems.

    The subtype holds the raw discovered step name and is used downstream by
    steps_by_video_to_message_content to format the aggregation prompt.
    """
    result = response.get("response", {})
    steps = result.get("steps", [])

    items = []
    for idx, step in enumerate(steps):
        name = step.get("name", "").strip().lower()
        content = step.get("content")
        start = step.get("start")
        end = step.get("end")
        items.append(KnowledgeItem(
            id=f"step_discovery:{name}-{idx}:{video.id}",
            type=KnowledgeItemType.STEP,
            text=content,
            source=KnowledgeSource(
                video_id=video.id,
                source_type="video",
                local_index=idx,
                time_range=(float(start), float(end)),
            ),
            subtype=None,
        ))
    return items


SYSTEM_MESSAGE_EXTRACTION = """You are an expert at extracting detailed procedural information from tutorial videos about the task: {task}. Given a taxonomy of steps, you can identify which steps appear in a video and provide comprehensive descriptions of how each step is performed, using both visual and verbal cues. Inputs are structured with XML tags — use the tag name and id attribute to identify each object and its boundaries."""

USER_MESSAGE_EXTRACTION = """You are given temporally aligned frames and transcript from a tutorial video and a taxonomy of steps. Extract detailed information about the steps from the taxonomy that appear in the video.

For each step that appears:
1. Confirm the step name from the taxonomy
2. Write a comprehensive natural text description of how this step is performed in the video
3. Provide the start and end times of the step in the video

Use visual frames to understand techniques and transcript for detailed instructions.
Make sure that steps have a clear start and end time in the video, and each segment of the video assigned to exactly one step.
Return ALL the task-relevant steps found in the video with detailed descriptions and start/end times.

INPUT:
- Taxonomy of steps:
{step_taxonomy}
- Tutorial:
{tutorial}
"""


def form_step_extraction_request_extraction(
    task: str,
    video: Video,
    use_vision: bool,
    chunk_config: ChunkConfig,
    step_taxonomy: List[Dict[str, Any]],
    **kwargs,
) -> Dict[str, Any]:
    """Form request to extract steps from a tutorial using a canonical taxonomy."""
    tutorial_content = video_to_message_content(video, use_vision, chunk_config)
    taxonomy_content = taxonomy_to_message_content(step_taxonomy)
    messages = [
        {
            "role": "system",
            "content": MessageContent.from_template(SYSTEM_MESSAGE_EXTRACTION, task=task),
        },
        {
            "role": "user",
            "content": MessageContent.from_template(
                USER_MESSAGE_EXTRACTION,
                step_taxonomy=taxonomy_content,
                tutorial=tutorial_content,
            ),
        },
    ]
    return {
        "messages": messages,
        "response_format": StepExtractionResult,
    }


def process_step_extraction_response_extraction(
    response: Dict[str, Any],
    video: Video,
    step_taxonomy: List[Dict[str, Any]],
    **kwargs,
) -> List[KnowledgeItem]:
    """Process step extraction response, returning typed KnowledgeItems."""
    result = response.get("response", {})
    steps = result.get("steps", [])

    labeled_segments = []

    valid_step_names = {step.get('name') for step in step_taxonomy}
    items = []
    for idx, step in enumerate(steps):
        name = step.get("name", "").strip().lower()
        if name not in valid_step_names:
            logger.warning(f"Step name {name} not in taxonomy! Skipping...")
            continue
        content = step.get("content")
        start = step.get("start")
        end = step.get("end")

        labeled_segments.append({
            "start": float(start),
            "end": float(end),
            "name": name,
        })
        items.append(KnowledgeItem(
            id=f"step:{name}-{idx}:{video.id}",
            type=KnowledgeItemType.STEP,
            text=content,
            source=KnowledgeSource(
                video_id=video.id,
                source_type="video",
                local_index=idx,
                time_range=(float(start), float(end)),
            ),
            subtype=name,
        ))

    ### check and raise warning if there are unlabeled transcript segments
    originally_unlabeled_duration = 0
    for i in range(len(video.transcript)):
        if i == 0:
            originally_unlabeled_duration += video.transcript[i].start
        if i == len(video.transcript) - 1:
            originally_unlabeled_duration += video.get_duration() - video.transcript[i].end
            continue
        originally_unlabeled_duration += video.transcript[i + 1].start - video.transcript[i].end
    
    labeled_segments = sorted(labeled_segments, key=lambda x: x["start"])
    unlabeled_segment_duration = 0
    for i in range(len(labeled_segments)):
        if labeled_segments[i]["start"] < 0 or labeled_segments[i]["end"] > video.get_duration():
            logger.warning(f"Invalid segment found for video {video.id}: {labeled_segments[i]['start']} - {labeled_segments[i]['end']}")
        if labeled_segments[i]["end"] < labeled_segments[i]["start"]:
            logger.warning(f"Invalid segment found for video {video.id}: {labeled_segments[i]['start']} - {labeled_segments[i]['end']}")
        if i == 0:
            unlabeled_segment_duration += labeled_segments[i]["start"]
        if i == len(labeled_segments) - 1:
            unlabeled_segment_duration += video.get_duration() - labeled_segments[i]["end"]
            continue
        if labeled_segments[i]["end"] > labeled_segments[i + 1]["start"]:
            logger.warning(f"Clashing segments found for video {video.id}: {labeled_segments[i]['start']} - {labeled_segments[i]['end']} and {labeled_segments[i + 1]['start']} - {labeled_segments[i + 1]['end']}")
        else:
            unlabeled_segment_duration += labeled_segments[i + 1]["start"] - labeled_segments[i]["end"]
    if abs(originally_unlabeled_duration - unlabeled_segment_duration) > 0:
        logger.warning(f"Unlabeled segment found for video {video.id}: {originally_unlabeled_duration - unlabeled_segment_duration} seconds")
    return items
