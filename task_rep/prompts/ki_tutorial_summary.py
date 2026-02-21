"""
Prompts for tutorial-level summarization.

Generates high-level summaries that capture the core procedural
knowledge of an entire tutorial video.
"""
from typing import Any, Dict, List

from utils.llm import MessageContent
from .ki_pydantic import TutorialInformation

SYSTEM_MESSAGE = """You are an expert at understanding and summarizing tutorial videos about the task: {task}. You excel at synthesizing information from both visual demonstrations and narration."""

USER_MESSAGE = """You are given temporally aligned frames and transcript from a tutorial video. Generate a comprehensive summary of the video using both visual context from frames and detailed information from the transcript that captures the core knowledge for completing the task.

Tutorial:
{tutorials}
"""

def form_tutorial_summary_request(
    task: str,
    tutorial_content: MessageContent,
    **kwargs,
) -> Dict[str, Any]:
    """Form request to extract tutorial summary from a video."""

    messages = [
        {
            "role": "system",
            "content": MessageContent.from_template(
                SYSTEM_MESSAGE,
                task=task
            )
        },
        {
            "role": "user",
            "content": MessageContent.from_template(
                USER_MESSAGE,
                tutorial=tutorial_content
            )
        }
    ]

    return {
        "messages": messages,
        "response_format": TutorialInformation,
    }

def process_tutorial_summary_response(
        response: Dict[str, Any],
        **kwargs,
    ) -> List[Dict[str, Any]]:
    """Process response from tutorial summary."""

    result = response.get("response", {})
    content = result.get("content", None)

    if not content:
        raise ValueError("No content found in response for tutorial summary")

    return [{
        "content": content,
    }]
