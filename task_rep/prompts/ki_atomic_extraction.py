"""
Prompts for atomic information extraction from tutorial videos.

Atomic information refers to individual pieces of knowledge that can be
classified into a taxonomy (e.g., methods, tips, context, explanations).
"""

from typing import List, Dict, Any

from utils.llm import MessageContent
from .formatters import taxonomy_to_str

from .ki_pydantic import AtomicExtractionResult

SYSTEM_MESSAGE = """You are an expert at analyzing tutorial videos and extracting atomic pieces of information about the task: {task}. You excel at identifying atomic information from both visual and audio content. You understand how to classify information using taxonomies and can accurately identify the start and end times of each piece of information."""

USER_MESSAGE = """You are given a taxonomy of atomic information and a tutorial with frames and transcript temporally aligned. Extract ALL atomic information from the tutorial.

For each atomic piece of information found in the tutorial:
1. Classify it using the taxonomy above (atomic_type)
2. Write a natural text description (content) that captures what is being communicated
3. Provide the start and end times of the atomic information in the tutorial

Use the visual frames to understand context and the transcript for detailed actions.
Make sure that atomic information have a clear start and end time in the tutorial.
Return ALL the atomic information found in the tutorial with their start and end times.

INPUT:
- Taxonomy of atomic information:
{taxonomy}
- Tutorial:
{tutorial}"""

def form_atomic_extraction_request(
    task: str,
    tutorial_content: MessageContent,
    information_taxonomy: List[Dict[str, Any]],
    **kwargs,
) -> Dict[str, Any]:
    """Form request to extract atomic information from a tutorial."""
    information_taxonomy_str = taxonomy_to_str(information_taxonomy)
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
                taxonomy=information_taxonomy_str,
                tutorial=tutorial_content
            )
        }
    ]
    return {
        "messages": messages,
        "response_format": AtomicExtractionResult,
    }

def process_atomic_extraction_response(
    response: Dict[str, Any],
    information_taxonomy: List[Dict[str, Any]],
    **kwargs,
) -> List[Dict[str, Any]]:
    """Process response from atomic extraction from a tutorial."""
    result = response.get("response", {})
    items = result.get("items", [])
    atomic_items = []

    valid_atomic_types = [info.get('name') for info in information_taxonomy]
    for item in items:
        atomic_type = item.get("atomic_type")
        if atomic_type not in valid_atomic_types:
            continue
        content = item.get("content")
        start = item.get("start")
        end = item.get("end")
        atomic_items.append({
            "atomic_type": atomic_type,
            "content": content,
            "start": float(start),
            "end": float(end),
        })
    return atomic_items