"""
Prompts for step-level extraction from tutorial videos.

This includes both step discovery (identifying what steps exist) and
step extraction based on a step taxonomy.
"""

from typing import Any, Dict, List
from pydantic import BaseModel, ConfigDict, Field

from task_rep.utils.llm import MessageContent


class StepInformation(BaseModel):
    """Information about a single step (used for both discovery and detailed extraction)."""
    model_config = ConfigDict(extra='forbid')

    name: str = Field(description="Name of this step (e.g., 'Prepare rice')")
    content: str = Field(description="Natural text content describing this step")
    start: float = Field(description="Start time of the step in the video")
    end: float = Field(description="End time of the step in the video")


class StepExtractionResult(BaseModel):
    """Result of step extraction from a video."""
    model_config = ConfigDict(extra='forbid')

    steps: List[StepInformation] = Field(description="All steps found in the video")


SYSTEM_MESSAGE_DISCOVERY = """You are an expert at analyzing tutorial videos and identifying procedural steps about the task: {task}. You excel at recognizing the high-level structure of tutorials, breaking down complex procedures into clear steps, and understanding both what is shown visually and what is explained in narration."""

USER_MESSAGE_DISCOVERY = """You are given temporally aligned frames and transcript from a tutorial video. Analyze the video and identify ALL procedural steps relevant to completing the task.

For each step:
1. Provide a short, clear step name
2. Write a comprehensive, but concise description of what this step involves
3. Provide the start and end times of the step in the video

Use both visual cues from frames and transcript information to identify steps.
Make sure that steps have a clear start and end time in the video, and each segment of the video assigned to exactly one step.
Return ALL the task-relevant steps found in the video with detailed descriptions and start/end times."""

def form_step_extraction_request_discovery(
    task: str,
    video_content: MessageContent,
    **kwargs,
) -> Dict[str, Any]:
    """Form request to extract steps from a video."""
    messages = [
        {
            "role": "system",
            "content": MessageContent(content=[
                {"type": "text", "text": SYSTEM_MESSAGE_DISCOVERY.format(task=task)}
            ])
        },
        {
            "role": "user",
            "content": MessageContent(content=[
                {"type": "text", "text": USER_MESSAGE_DISCOVERY},
                *video_content.to_content()
            ])
        }
    ]
    return {
        "messages": messages,
        "response_format": StepExtractionResult,
    }

def process_step_extraction_response_discovery(
    response: Dict[str, Any],
    **kwargs,
) -> List[Dict[str, Any]]:
    """Process response from step extraction discovery."""
    result = response.get("response", {})
    steps = result.get("steps", [])

    step_dicts = []
    for step in steps:
        name = step.get("name")
        content = step.get("content")
        start = step.get("start")
        end = step.get("end")
        step_dicts.append({
            "name": name,
            "content": content,
            "start": float(start),
            "end": float(end),
        })
    return step_dicts

SYSTEM_MESSAGE_EXTRACTION = """You are an expert at extracting detailed procedural information from tutorial videos about the task: {task}. Given a taxonomy of steps, you can identify which steps appear in a video and provide comprehensive descriptions of how each step is performed, using both visual and verbal cues."""

USER_MESSAGE_EXTRACTION = """You are given temporally aligned frames and transcript from a tutorial video. Extract detailed information about ALL the following steps from the video.

Steps to extract (only include steps that actually appear in this video):
{step_taxonomy}

For each step that appears:
1. Confirm the step name from the list above
2. Write a comprehensive natural text description of how this step is performed in the video
3. Provide the start and end times of the step in the video

Use visual frames to understand techniques and transcript for detailed instructions.
Make sure that steps have a clear start and end time in the video, and each segment of the video assigned to exactly one step.
Return ALL the task-relevant steps found in the video with detailed descriptions and start/end times."""


def form_step_extraction_request_extraction(
    task: str,
    video_content: MessageContent,
    step_taxonomy: List[Dict[str, Any]],
    **kwargs,
) -> Dict[str, Any]:
    """Form request to extract steps from a video."""

    step_taxonomy_text = "\n".join([f"- {step.get('name')}: {step.get('definition')}" for step in step_taxonomy])
    messages = [
        {
            "role": "system",
            "content": MessageContent(content=[
                {"type": "text", "text": SYSTEM_MESSAGE_EXTRACTION.format(task=task)}
            ])
        },
        {
            "role": "user",
            "content": MessageContent(content=[
                {"type": "text", "text": USER_MESSAGE_EXTRACTION.format(step_taxonomy=step_taxonomy_text)},
                *video_content.to_content()
            ])
        }
    ]
    return {
        "messages": messages,
        "response_format": StepExtractionResult,
    }

def process_step_extraction_response_extraction(
    response: Dict[str, Any],
    step_taxonomy: List[Dict],
    **kwargs,
) -> List[Dict[str, Any]]:
    """Process response from step extraction."""
    result = response.get("response", {})
    steps = result.get("steps", [])

    ### check if the step is in the step taxonomy
    valid_step_names = [step['name'] for step in step_taxonomy]
    
    step_dicts = []
    for step in steps:
        name = step.get("name")
        if name not in valid_step_names:
            continue
        content = step.get("content")
        start = step.get("start")
        end = step.get("end")
        step_dicts.append({
            "name": name,
            "content": content,
            "start": float(start),
            "end": float(end),
        })
    return step_dicts