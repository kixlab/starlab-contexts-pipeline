"""
Prompts for tutorial-level summarization.

Generates high-level summaries that capture the core procedural
knowledge of an entire tutorial video.
"""
from pydantic import BaseModel, ConfigDict, Field
from typing import Any, Dict, List
from task_rep.utils.llm import MessageContent


class TutorialInformation(BaseModel):
    """High-level tutorial summary."""
    model_config = ConfigDict(extra='forbid')

    content: str = Field(description="Comprehensive summary of the entire tutorial")


SYSTEM_MESSAGE = """You are an expert at understanding and summarizing tutorial videos about the task: {task}. You excel at synthesizing information from both visual demonstrations and narration."""

USER_MESSAGE = """You are given temporally aligned frames and transcript from a tutorial video. Generate a comprehensive summary of the video using both visual context from frames and detailed information from the transcript that captures the core knowledge for completing the task."""

def form_tutorial_summary_request(
    task: str,
    video_content: MessageContent,
    **kwargs,
) -> Dict[str, Any]:
    """Form request to extract tutorial summary from a video."""

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
                {"type": "text", "text": USER_MESSAGE},
                *video_content.to_content()
            ])
        }
    ]

    return {
        "messages": messages,
        "response_format": TutorialInformation,
    }

def process_tutorial_summary_response(
        response: Dict[str, Any],
        **kwargs,
    ) -> List[Any]:
    """Process response from tutorial summary."""

    result = response.get("response", {})
    content = result.get("content", None)

    if not content:
        raise ValueError("No content found in response for tutorial summary")

    return [{
        "content": content,
    }]
