"""
Prompts for atomic information extraction from tutorial videos.

Atomic information refers to individual pieces of knowledge that can be
classified into a taxonomy (e.g., methods, tips, context, explanations).
"""

from pydantic import BaseModel, ConfigDict, Field
from typing import List, Dict, Any

from task_rep.utils.llm import MessageContent

class AtomicInformation(BaseModel):
    """Single atomic piece of information with taxonomy classification."""
    model_config = ConfigDict(extra='forbid')

    atomic_type: str = Field(description="Classification from taxonomy (e.g., 'Method - Instruction')")
    content: str = Field(description="Natural text content of this atomic information")
    start: float = Field(description="Start time of the atomic information in the video")
    end: float = Field(description="End time of the atomic information in the video")


class AtomicExtractionResult(BaseModel):
    """Result of extracting all atomic information from a video."""
    model_config = ConfigDict(extra='forbid')

    items: List[AtomicInformation] = Field(description="All atomic information items found in the video")


SYSTEM_MESSAGE = """You are an expert at analyzing tutorial videos and extracting atomic pieces of information about the task: {task}. You excel at identifying atomic information from both visual and audio content. You understand how to classify information using taxonomies and can accurately identify the start and end times of each piece of information."""

USER_MESSAGE = """Extract ALL atomic information from this tutorial video.

The video content is provided below with frames and transcript temporally aligned.

INFORMATION TAXONOMY (classify each item):
{information_taxonomy}

For each atomic piece of information found in the video:
1. Classify it using the taxonomy above (atomic_type)
2. Write a natural text description (content) that captures what is being communicated
3. Provide the start and end times of the atomic information in the video

Use the visual frames to understand context and the transcript for detailed actions.
Make sure that atomic information have a clear start and end time in the video.
Return ALL the atomic information found in the video with their start and end times."""

def form_atomic_extraction_request(
    task: str,
    video_content: MessageContent,
    information_taxonomy: List[Dict[str, Any]],
    **kwargs,
) -> Dict[str, Any]:
    """Form request to extract atomic information from a video."""
    information_taxonomy_text = "\n".join([f"- {info['name']}: {info['definition']}" for info in information_taxonomy])
    messages = [
        {
            "role": "system",
            "content": MessageContent(content=[
                {"type": "text", "text": SYSTEM_MESSAGE.format(task=task)}
            ])
        },
        {
            "role": "user",
            "content": MessageContent(content=[
                {"type": "text", "text": USER_MESSAGE.format(information_taxonomy=information_taxonomy_text)},
                *video_content.to_content()
            ])
        }
    ]
    return {
        "messages": messages,
        "response_format": AtomicExtractionResult,
    }

def process_atomic_extraction_response(
    response: Dict[str, Any],
    **kwargs,
) -> List[AtomicInformation]:
    """Process response from atomic extraction."""
    result = response.get("response", {})
    items = result.get("items", [])
    atomic_items = []
    for item in items:
        atomic_type = item.get("atomic_type")
        content = item.get("content")
        start = item.get("start")
        end = item.get("end")
        atomic_items.append({
            "atomic_type": atomic_type,
            "content": content,
            "start": start,
            "end": end,
        })
    return atomic_items