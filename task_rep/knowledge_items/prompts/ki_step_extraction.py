"""
Prompts for step-level extraction from tutorial videos.

Step discovery: identify the procedural steps a video shows.
"""

import logging
from typing import Any, Dict, List

from utils import MessageContent

from .formatters import video_to_message_content
from .ki_pydantic import StepExtractionResult

from knowledge_items.schema import KnowledgeItem, KnowledgeItemType, KnowledgeSource, Video
from knowledge_items.chunk_sampling import ChunkConfig

logger = logging.getLogger(__name__)


SYSTEM_MESSAGE_DISCOVERY = """You are an expert at analyzing tutorial videos and identifying procedural steps about the task: {task}. You excel at recognizing the high-level structure of tutorials, breaking down complex procedures into clear steps, and understanding both what is shown visually and what is explained in narration. You focus ONLY on steps that are directly part of performing the task itself — not on video production, presentation, or meta-commentary. Inputs are structured with XML tags — use the tag name and id attribute to identify each object and its boundaries."""

USER_MESSAGE_DISCOVERY = """You are given temporally aligned frames and transcript from a tutorial video. Analyze the video and identify ALL procedural steps relevant to completing the task.

For each step:
1. Provide a short, clear step name
2. Write a comprehensive, but concise description of what this step involves
3. Provide the start and end times of the step in the video

IMPORTANT:
- Only include steps that are directly part of performing the task.
- Do NOT include generic video segments such as introductions, conclusions, sponsor messages, channel promotions, personal anecdotes, equipment setup for filming, or other content not directly related to performing the task.
- If a segment contains both task-relevant and task-irrelevant content, extract only the task-relevant portion and adjust the time boundaries accordingly.

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
