"""
Prompts for tutorial-level summarization.

Generates high-level summaries that capture the core procedural
knowledge of an entire tutorial video.
"""

from typing import Any, Dict, List

from utils import MessageContent

from .formatters import video_to_message_content
from .ki_pydantic import TutorialInformation

from knowledge_items.schema import KnowledgeItem, KnowledgeItemType, KnowledgeSource, Video
from knowledge_items.chunk_sampling import ChunkConfig


SYSTEM_MESSAGE = """You are an expert at understanding and summarizing tutorial videos about the task: {task}. You excel at synthesizing information from both visual demonstrations and narration."""

USER_MESSAGE = """You are given temporally aligned frames and transcript from a tutorial video. Generate a comprehensive summary of the video using both visual context from frames and detailed information from the transcript that captures the core knowledge for completing the task.

Tutorial:
{tutorial}
"""


def form_tutorial_summary_request(
    task: str,
    video: Video,
    use_vision: bool,
    chunk_config: ChunkConfig,
    **kwargs,
) -> Dict[str, Any]:
    """Form request to summarize a tutorial video."""
    tutorial_content = video_to_message_content(video, use_vision, chunk_config)
    messages = [
        {
            "role": "system",
            "content": MessageContent.from_template(SYSTEM_MESSAGE, task=task),
        },
        {
            "role": "user",
            "content": MessageContent.from_template(
                USER_MESSAGE,
                tutorial=tutorial_content,
            ),
        },
    ]
    return {
        "messages": messages,
        "response_format": TutorialInformation,
    }


def process_tutorial_summary_response(
    response: Dict[str, Any],
    video: Video,
    **kwargs,
) -> List[KnowledgeItem]:
    """Process tutorial summary response, returning a single KnowledgeItem."""
    result = response.get("response", {})
    content = result.get("content")

    if not content:
        raise ValueError("No content found in response for tutorial summary")

    return [KnowledgeItem(
        id=f"tutorial:{video.id}",
        type=KnowledgeItemType.TUTORIAL,
        text=content,
        source=KnowledgeSource(
            video_id=video.id,
            source_type="video",
            local_index=0,
            time_range=(0, video.get_duration()),
        ),
        subtype=None,
    )]
