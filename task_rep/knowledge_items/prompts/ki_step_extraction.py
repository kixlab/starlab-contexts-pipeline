"""
Prompts for step-level extraction from tutorial videos.

This includes both step discovery (identifying what steps exist) and
step extraction based on a step taxonomy.
"""

from typing import Any, Dict, List

from utils import MessageContent

from .formatters import taxonomy_to_message_content, video_to_message_content
from .ki_pydantic import StepExtractionResult

from knowledge_items.schema import KnowledgeItem, KnowledgeItemType, KnowledgeSource, Video
from knowledge_items.chunk_sampling import ChunkConfig


SYSTEM_MESSAGE_DISCOVERY = """You are an expert at analyzing tutorial videos and identifying procedural steps about the task: {task}. You excel at recognizing the high-level structure of tutorials, breaking down complex procedures into clear steps, and understanding both what is shown visually and what is explained in narration."""

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
            subtype=name,
        ))
    return items


SYSTEM_MESSAGE_EXTRACTION = """You are an expert at extracting detailed procedural information from tutorial videos about the task: {task}. Given a taxonomy of steps, you can identify which steps appear in a video and provide comprehensive descriptions of how each step is performed, using both visual and verbal cues."""

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

    valid_step_names = {step.get('name') for step in step_taxonomy}
    items = []
    for idx, step in enumerate(steps):
        name = step.get("name", "").strip().lower()
        if name not in valid_step_names:
            continue
        content = step.get("content")
        start = step.get("start")
        end = step.get("end")
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
    return items
