"""
Simple example: Create a task and build a dataset.
"""

from task_rep import TaskManager, build_dataset


def example_create_task():
    """Create a task in tasks.json."""
    manager = TaskManager()

    # Create a task with just video IDs
    task = manager.create_task(
        task_id="sushi_v1",
        name="How to Make a Sushi Roll",
        video_ids=["ZXT4XLV3j4A", "abc123xyz", "def456"],
        metadata={"description": "Tutorial videos on making sushi"}
    )

    print(f"✓ Created task: {task.id}")
    print(f"  Videos: {len(task.video_ids)}")


def example_build_dataset():
    """Build a dataset from task definition."""
    # This will:
    # 1. Download videos to static/videos/{video_id}/
    # 2. Extract frames at 1 fps
    # 3. Fetch transcripts (Whisper by default)
    # 4. Create static/datasets/sushi_v1.json

    output = build_dataset(
        task_id="sushi_v1",
        fps=1.0,
        transcript_source="whisper"  # or "youtube-or-whisper"
    )

    print(f"✓ Dataset built: {output}")


def example_use_dataset():
    """Use the dataset for extraction."""
    import json
    from task_rep import VLMClient, MessageContent
    from pydantic import BaseModel, Field

    class KnowledgeExtraction(BaseModel):
        actions: list[str] = Field(description="Actions in segment")
        objects: list[str] = Field(description="Key objects")

    # Load dataset
    with open("static/datasets/sushi_v1.json") as f:
        dataset = json.load(f)

    client = VLMClient(provider="openai")

    # Extract from first video
    video = dataset[0]
    if "transcript" in video:
        for segment in video["transcript"][:3]:  # First 3 segments
            # Get frames for this segment
            segment_frames = [
                f["path"] for f in video["frames"]
                if segment["start"] <= f["time"] <= segment["end"]
            ][:4]

            if not segment_frames:
                continue

            # Extract with VLM
            response = client.generate(
                messages=[{
                    "role": "user",
                    "content": MessageContent(
                        content=[
                            {"type": "text", "text": f"Extract knowledge: {segment['text']}"}
                        ] + [{"type": "image_path", "path": p} for p in segment_frames]
                    )
                }],
                response_format=KnowledgeExtraction
            )

            print(f"Segment: {segment['text'][:50]}...")
            print(f"  Actions: {response['actions']}")
            print(f"  Objects: {response['objects']}")
            print()


if __name__ == "__main__":
    # Step 1: Create task (run once)
    # example_create_task()

    # Step 2: Build dataset (downloads videos, extracts frames, fetches transcripts)
    # example_build_dataset()

    # Step 3: Use the dataset
    # example_use_dataset()

    print("Examples ready. Uncomment to run.")
