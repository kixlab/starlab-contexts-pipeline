"""
Prompts for tutorial-level summarization.

Generates high-level summaries that capture the core procedural
knowledge of an entire tutorial video.
"""


def get_system_message() -> str:
    """Get the system message for tutorial summarization."""
    return """You are an expert at understanding and summarizing tutorial videos. You excel at synthesizing information from both visual demonstrations and narration to create comprehensive summaries that capture the core procedural knowledge."""


def get_instruction() -> str:
    """
    Get user instruction for tutorial summarization.

    Returns:
        Instruction text for the user message
    """
    return """Create a comprehensive summary of this tutorial video.

The video content is provided below with frames and transcript temporally aligned.

Write a natural text summary that captures the core procedural knowledge of the tutorial.
Use both visual context from frames and detailed information from the transcript.
Make it comprehensive but concise.

Return a natural text summary of the tutorial."""
