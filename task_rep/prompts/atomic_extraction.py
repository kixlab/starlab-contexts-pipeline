"""
Prompts for atomic information extraction from tutorial videos.

Atomic information refers to individual pieces of knowledge that can be
classified into a taxonomy (e.g., methods, tips, context, explanations).
"""


def get_system_message() -> str:
    """Get the system message for atomic extraction."""
    return """You are an expert at analyzing tutorial videos and extracting atomic pieces of information. You excel at identifying individual facts, instructions, tips, and contextual details from both visual and audio content. You understand how to classify information using taxonomies and can accurately identify the start and end times of each piece of information."""


def get_instruction(taxonomy_desc: str) -> str:
    """
    Get user instruction for atomic extraction.

    Args:
        taxonomy_desc: Formatted string describing the taxonomy categories

    Returns:
        Instruction text for the user message
    """
    return f"""Extract ALL atomic information from this tutorial video.

The video content is provided below with frames and transcript temporally aligned.

TAXONOMY (classify each item):
{taxonomy_desc}

For each atomic piece of information:
1. Classify it using the taxonomy above (atomic_type)
2. Write a natural text description (content) that captures what is being communicated

Use the visual frames to understand context and the transcript for detailed actions.
Return ALL atomic information found in the video."""
