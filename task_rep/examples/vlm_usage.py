"""
Examples demonstrating VLM client usage with and without images.
"""

import asyncio
from pydantic import BaseModel, Field
from typing import List

from task_rep import VLMClient, MessageContent, get_vlm_client
from task_rep.utils.llm import (
    llm_enqueue_request,
    llm_get_responses,
    process_api_requests,
)


# ============== Example 1: Simple text-only generation ==============


class StepExtraction(BaseModel):
    """Example Pydantic model for step extraction."""

    steps: List[str] = Field(description="List of procedural steps")


def example_text_only():
    """Generate structured output from text."""
    client = get_vlm_client(provider="openai", model="gpt-4.1-mini-2025-04-14")

    response = client.generate(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "Extract the steps from this text: First, preheat the oven. Then, mix the ingredients. Finally, bake for 30 minutes.",
            },
        ],
        response_format=StepExtraction,
    )

    print("Text-only response:", response)


# ============== Example 2: Single image analysis ==============


class FrameDescription(BaseModel):
    """Example model for frame description."""

    description: str = Field(description="Detailed description of what's in the frame")
    objects: List[str] = Field(description="List of visible objects")


def example_single_image():
    """Analyze a single image."""
    client = get_vlm_client(provider="openai")

    response = client.generate(
        messages=[
            {"role": "system", "content": "You are a vision assistant."},
            {
                "role": "user",
                "content": MessageContent(
                    text="Describe this image in detail.",
                    image_paths=["path/to/frame.jpg"],
                ),
            },
        ],
        response_format=FrameDescription,
    )

    print("Single image response:", response)


# ============== Example 3: Multiple images (video frames) ==============


class VideoActionDescription(BaseModel):
    """Example model for action description from multiple frames."""

    action: str = Field(description="The action being performed")
    key_objects: List[str] = Field(description="Key objects involved")
    temporal_progression: str = Field(
        description="How the action progresses across frames"
    )


def example_multiple_images():
    """Analyze multiple video frames together."""
    client = get_vlm_client(provider="openai")

    response = client.generate(
        messages=[
            {"role": "system", "content": "You are a video analysis assistant."},
            {
                "role": "user",
                "content": MessageContent(
                    text="What action is being performed across these frames?",
                    image_paths=[
                        "path/to/frame_001.jpg",
                        "path/to/frame_002.jpg",
                        "path/to/frame_003.jpg",
                    ],
                ),
            },
        ],
        response_format=VideoActionDescription,
    )

    print("Multiple images response:", response)


# ============== Example 4: Images with raw bytes ==============


def example_image_bytes():
    """Use image bytes directly (e.g., from in-memory processing)."""
    client = get_vlm_client(provider="openai")

    # Simulate reading image bytes
    with open("path/to/frame.jpg", "rb") as f:
        image_data = f.read()

    response = client.generate(
        messages=[
            {
                "role": "user",
                "content": MessageContent(
                    text="What do you see?",
                    image_bytes=[image_data],
                ),
            }
        ],
        response_format=FrameDescription,
    )

    print("Image bytes response:", response)


# ============== Example 5: Batch processing with images ==============


class KnowledgeExtraction(BaseModel):
    """Example model for knowledge extraction."""

    knowledge_items: List[str] = Field(
        description="Atomic knowledge items extracted from the video segment"
    )


async def example_batch_with_images():
    """Process multiple video segments in parallel with frame images."""
    batch_id = "tutorial_extraction_001"

    # Enqueue requests for multiple video segments
    video_segments = [
        {
            "text": "In this part, we prepare the ingredients...",
            "frames": ["frames/seg1_frame1.jpg", "frames/seg1_frame2.jpg"],
        },
        {
            "text": "Now we mix everything together...",
            "frames": ["frames/seg2_frame1.jpg", "frames/seg2_frame2.jpg"],
        },
        {
            "text": "Finally, we bake it in the oven...",
            "frames": ["frames/seg3_frame1.jpg", "frames/seg3_frame2.jpg"],
        },
    ]

    request_ids = []
    for i, segment in enumerate(video_segments):
        messages = [
            {"role": "system", "content": "Extract knowledge from video segments."},
            {
                "role": "user",
                "content": MessageContent(
                    text=f"Extract atomic knowledge items from this segment: {segment['text']}",
                    image_paths=segment["frames"],
                ).to_openai_content(),  # Convert to OpenAI format
            },
        ]

        req_id = llm_enqueue_request(
            batch_id=batch_id,
            messages=messages,
            response_format=KnowledgeExtraction,
            model="gpt-4.1-mini-2025-04-14",
            request_id=f"segment_{i}",
        )
        request_ids.append(req_id)

    # Process all requests in parallel
    await process_api_requests(batch_id=batch_id, max_concurrency=10, provider="openai")

    # Retrieve all responses
    responses = llm_get_responses(batch_id)

    for req_id in request_ids:
        print(f"Response for {req_id}:", responses[req_id])


# ============== Example 6: Switching between OpenAI and Anthropic ==============


def example_provider_switching():
    """Demonstrate using different providers."""

    # OpenAI client
    openai_client = get_vlm_client(provider="openai")
    openai_response = openai_client.generate(
        messages=[
            {
                "role": "user",
                "content": MessageContent(
                    text="Describe this frame.",
                    image_paths=["frame.jpg"],
                ),
            }
        ],
        response_format=FrameDescription,
    )

    # Anthropic/Bedrock client
    anthropic_client = get_vlm_client(
        provider="anthropic", model="us.anthropic.claude-3-5-haiku-20241022-v1:0"
    )
    anthropic_response = anthropic_client.generate(
        messages=[
            {
                "role": "user",
                "content": MessageContent(
                    text="Describe this frame.",
                    image_paths=["frame.jpg"],
                ),
            }
        ],
        response_format=FrameDescription,
    )

    print("OpenAI:", openai_response)
    print("Anthropic:", anthropic_response)


# ============== Example 7: Extract knowledge from video with frames ==============


class TutorialKnowledge(BaseModel):
    """Knowledge extracted from a tutorial video segment."""

    main_action: str = Field(description="The main action being performed")
    sub_steps: List[str] = Field(description="Sub-steps of the action")
    visible_tools: List[str] = Field(description="Tools visible in the frames")
    context: str = Field(description="Contextual information")


def example_tutorial_extraction():
    """Extract structured knowledge from a tutorial segment with frames."""
    from task_rep import Video, VideoTranscriptSegment, VideoFrame

    # Simulate a video with transcript and frames
    video = Video(
        id="tutorial_001",
        url="https://example.com/video.mp4",
        title="How to Make Pizza",
        original_title="How to Make Pizza",
        content="Learn to make authentic Italian pizza from scratch.",
        transcript=[
            VideoTranscriptSegment(
                text="First, we need to prepare our dough. Take the flour and...",
                start=0.0,
                end=10.5,
            ),
            VideoTranscriptSegment(
                text="Mix it with water, yeast, and a pinch of salt.",
                start=10.5,
                end=18.2,
            ),
        ],
        frames=[
            VideoFrame(path="frames/t001_00.jpg", caption="", time=5.0),
            VideoFrame(path="frames/t001_01.jpg", caption="", time=15.0),
        ],
    )

    # Use VLM to extract knowledge from a segment with visual context
    client = get_vlm_client(provider="openai")

    segment = video.transcript[0]
    # Find frames within the segment time range
    segment_frames = [
        f.path for f in video.frames if segment.start <= f.time <= segment.end
    ]

    response = client.generate(
        messages=[
            {
                "role": "system",
                "content": "You are an expert at extracting structured knowledge from tutorial videos.",
            },
            {
                "role": "user",
                "content": MessageContent(
                    text=f"Extract knowledge from this tutorial segment: '{segment.text}'",
                    image_paths=segment_frames,
                ),
            },
        ],
        response_format=TutorialKnowledge,
    )

    print("Tutorial knowledge:", response)


if __name__ == "__main__":
    # Run synchronous examples
    print("=== Example 1: Text only ===")
    # example_text_only()

    print("\n=== Example 2: Single image ===")
    # example_single_image()

    print("\n=== Example 3: Multiple images ===")
    # example_multiple_images()

    print("\n=== Example 5: Batch with images ===")
    # asyncio.run(example_batch_with_images())

    print("\n=== Example 6: Provider switching ===")
    # example_provider_switching()

    print("\n=== Example 7: Tutorial extraction ===")
    # example_tutorial_extraction()

    print("\nAll examples defined. Uncomment to run.")
