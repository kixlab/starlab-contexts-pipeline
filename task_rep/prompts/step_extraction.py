"""
Prompts for step-level extraction from tutorial videos.

This includes both step discovery (identifying what steps exist) and
step extraction (getting detailed descriptions of specific steps).
"""


def get_system_message(phase: str = "discovery") -> str:
    """
    Get system message for step extraction.

    Args:
        phase: Either "discovery" or "extraction"

    Returns:
        System message text
    """
    if phase == "discovery":
        return """You are an expert at analyzing tutorial videos and identifying procedural steps. You excel at recognizing the high-level structure of tutorials, breaking down complex procedures into clear steps, and understanding both what is shown visually and what is explained in narration."""
    elif phase == "extraction":
        return """You are an expert at extracting detailed procedural information from tutorial videos. Given a taxonomy of steps, you can identify which steps appear in a video and provide comprehensive descriptions of how each step is performed, using both visual and verbal cues."""
    else:
        raise ValueError(f"Invalid phase: {phase}. Must be 'discovery' or 'extraction'")


def get_instruction(phase: str = "discovery", step_taxonomy: list[str] = None) -> str:
    """
    Get user instruction for step extraction.

    Args:
        phase: Either "discovery" or "extraction"
        step_taxonomy: List of canonical step names (required if phase="extraction")

    Returns:
        Instruction text for the user message
    """
    if phase == "discovery":
        return """Analyze this tutorial video and identify ALL procedural steps relevant to completing the task.

The video content is provided below with frames and transcript temporally aligned.

For each step:
1. Provide a short, clear step name
2. Write a natural text description of what this step involves

Use both visual cues from frames and transcript information to identify steps.
Return ALL the task-relevant steps found in the video."""

    elif phase == "extraction":
        if not step_taxonomy:
            raise ValueError("step_taxonomy is required for extraction phase")

        step_list = "\n".join([f"- {step}" for step in step_taxonomy])
        return f"""Extract detailed information about ALL the following steps from this video.

The video content is provided below with frames and transcript temporally aligned.

Steps to extract (only include steps that actually appear in this video):
{step_list}

For each step that appears:
1. Confirm the step name from the list above
2. Write a comprehensive natural text description of how this step is performed in this specific video

Use visual frames to understand techniques and transcript for detailed instructions.
Try to describe each step comprehensively and accurately based on the video content."""

    else:
        raise ValueError(f"Invalid phase: {phase}. Must be 'discovery' or 'extraction'")
