"""
Example: End-to-end multimodal knowledge extraction from tutorial videos.

This example demonstrates:
1. Preparing a dataset with video frames
2. Extracting knowledge items using VLMs with visual context
3. Saving the enhanced knowledge base
"""

import asyncio
import json
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List

from task_rep import (
    VLMClient,
    MessageContent,
    KnowledgeItem,
    KnowledgeItemType,
    KnowledgeSource,
    KnowledgeSubtype,
    llm_enqueue_request,
    llm_get_responses,
    process_api_requests,
    build_dataset,
)


# ============== Step 1: Prepare Dataset with Frames ==============


def prepare_dataset(task_id: str):
    """Download videos, extract frames, and fetch transcripts."""
    print(f"=== Preparing dataset for: {task_id} ===\n")

    output_path = build_dataset(
        task_id=task_id,
        fps=1.0,  # 1 frame per second
        transcript_source="openai",
    )

    return output_path


# ============== Step 2: Define Extraction Schema ==============


class SegmentKnowledge(BaseModel):
    """Knowledge extracted from a video segment with visual context."""

    actions: List[str] = Field(
        description="Actions being performed in this segment"
    )
    objects: List[str] = Field(
        description="Key objects visible in the frames"
    )
    context: str = Field(
        description="Contextual information about the setting and method"
    )
    tips: List[str] = Field(
        description="Tips or warnings mentioned in this segment",
        default_factory=list,
    )


# ============== Step 3: Extract Knowledge with VLM ==============


def extract_knowledge_single(task: str, dataset_path: str, max_segments: int = 10):
    """Extract knowledge from dataset using VLM (synchronous, for testing)."""
    print(f"\n=== Extracting knowledge for: {task} ===\n")

    # Load dataset with frames
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    client = VLMClient(provider="openai", model="gpt-4.1-mini-2025-04-14")

    knowledge_items = []
    segment_count = 0

    for video in dataset:
        if not video.get("frames"):
            print(f"Skipping video {video['id']} (no frames)")
            continue

        print(f"\nProcessing video: {video['id']}")

        # Process each transcript segment
        for seg_idx, segment in enumerate(video["transcript"]):
            if segment_count >= max_segments:
                break

            # Find frames within this segment's time range
            segment_frames = [
                f["path"]
                for f in video["frames"]
                if segment["start"] <= f["time"] <= segment["end"]
            ]

            # Skip segments with no frames
            if not segment_frames:
                continue

            # Limit frames per segment (max 3-4 for token efficiency)
            segment_frames = segment_frames[:4]

            print(f"  Segment {seg_idx}: {len(segment_frames)} frames")

            try:
                # Extract with visual context
                response = client.generate(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert at extracting structured knowledge from tutorial videos.",
                        },
                        {
                            "role": "user",
                            "content": MessageContent(
                                content=[
                                    {"type": "text", "text": f"Extract knowledge from this tutorial segment:\n\nTranscript: {segment['text']}"}
                                ] + [{"type": "image_path", "path": p} for p in segment_frames]
                            ),
                        },
                    ],
                    response_format=SegmentKnowledge,
                )

                # Convert to knowledge items
                for action in response["actions"]:
                    item = KnowledgeItem(
                        id=f"{video['id']}:seg{seg_idx}:action{len(knowledge_items)}",
                        type=KnowledgeItemType.ATOMIC,
                        text=action,
                        source=KnowledgeSource(
                            video_id=video["id"],
                            local_index=seg_idx,
                            time_range=(segment["start"], segment["end"]),
                        ),
                        subtype=KnowledgeSubtype.METHOD_INSTRUCTION,
                        metadata={
                            "video_title": video["original_title"],
                            "objects": response["objects"],
                            "context": response["context"],
                            "frame_paths": segment_frames,
                        },
                    )
                    knowledge_items.append(item)

                segment_count += 1

            except Exception as e:
                print(f"  Error extracting from segment {seg_idx}: {e}")
                continue

        if segment_count >= max_segments:
            break

    print(f"\n✓ Extracted {len(knowledge_items)} knowledge items")
    return knowledge_items


# ============== Step 4: Batch Extraction (Parallel) ==============


async def extract_knowledge_batch(
    task: str, dataset_path: str, max_segments: int = 50
):
    """Extract knowledge in parallel using batch processing."""
    print(f"\n=== Batch extracting knowledge for: {task} ===\n")

    # Load dataset
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    batch_id = f"extraction_{task.replace(' ', '_').lower()}"

    # Enqueue all requests
    request_metadata = []
    segment_count = 0

    for video in dataset:
        if not video.get("frames"):
            continue

        for seg_idx, segment in enumerate(video["transcript"]):
            if segment_count >= max_segments:
                break

            segment_frames = [
                f["path"]
                for f in video["frames"]
                if segment["start"] <= f["time"] <= segment["end"]
            ]

            if not segment_frames:
                continue

            segment_frames = segment_frames[:4]

            # Build message with images
            message_content = MessageContent(
                content=[
                    {"type": "text", "text": f"Extract knowledge from this tutorial segment:\n\nTranscript: {segment['text']}"}
                ] + [{"type": "image_path", "path": p} for p in segment_frames]
            )

            messages = [
                {
                    "role": "system",
                    "content": "You are an expert at extracting structured knowledge from tutorial videos.",
                },
                {"role": "user", "content": message_content.to_openai_content()},
            ]

            req_id = llm_enqueue_request(
                batch_id=batch_id,
                messages=messages,
                response_format=SegmentKnowledge,
                model="gpt-4.1-mini-2025-04-14",
                request_id=f"{video['id']}_seg{seg_idx}",
            )

            request_metadata.append(
                {
                    "request_id": req_id,
                    "video_id": video["id"],
                    "segment_index": seg_idx,
                    "segment": segment,
                    "frame_paths": segment_frames,
                    "video_title": video["original_title"],
                }
            )

            segment_count += 1

        if segment_count >= max_segments:
            break

    print(f"Enqueued {len(request_metadata)} requests")

    # Process in parallel
    await process_api_requests(
        batch_id=batch_id,
        max_concurrency=32,
        provider="openai",
    )

    # Retrieve results and build knowledge items
    responses = llm_get_responses(batch_id)

    knowledge_items = []
    for meta in request_metadata:
        req_id = meta["request_id"]
        if req_id not in responses:
            continue

        response = responses[req_id]

        # Create knowledge items from actions
        for action in response["actions"]:
            item = KnowledgeItem(
                id=f"{meta['video_id']}:seg{meta['segment_index']}:action{len(knowledge_items)}",
                type=KnowledgeItemType.ATOMIC,
                text=action,
                source=KnowledgeSource(
                    video_id=meta["video_id"],
                    local_index=meta["segment_index"],
                    time_range=(
                        meta["segment"]["start"],
                        meta["segment"]["end"],
                    ),
                ),
                subtype=KnowledgeSubtype.METHOD_INSTRUCTION,
                metadata={
                    "video_title": meta["video_title"],
                    "objects": response["objects"],
                    "context": response["context"],
                    "frame_paths": meta["frame_paths"],
                },
            )
            knowledge_items.append(item)

    print(f"\n✓ Extracted {len(knowledge_items)} knowledge items")
    return knowledge_items


# ============== Step 5: Save Knowledge Base ==============


def save_knowledge_base(knowledge_items: List[KnowledgeItem], task_id: str):
    """Save extracted knowledge items."""
    output_dir = Path(f"static/datasets")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{task_id}_knowledge_items.jsonl"

    with open(output_path, "w", encoding="utf-8") as f:
        for item in knowledge_items:
            f.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")

    print(f"\n✓ Saved knowledge base to: {output_path}")
    return str(output_path)


# ============== Main Pipeline ==============


def run_pipeline_sync(task_id: str, max_segments: int = 10):
    """Run the full pipeline synchronously (for testing)."""
    print(f"{'='*60}")
    print(f"MULTIMODAL EXTRACTION PIPELINE: {task_id}")
    print(f"{'='*60}\n")

    # Step 1: Prepare dataset with frames and transcripts
    dataset_path = prepare_dataset(task_id)

    # Step 2: Extract knowledge with VLM
    knowledge_items = extract_knowledge_single(
        task_id, dataset_path, max_segments=max_segments
    )

    # Step 3: Save knowledge base
    output_path = save_knowledge_base(knowledge_items, task_id)

    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"Dataset: {dataset_path}")
    print(f"Knowledge base: {output_path}")
    print(f"Total items: {len(knowledge_items)}")


async def run_pipeline_batch(task_id: str, max_segments: int = 50):
    """Run the full pipeline with batch processing."""
    print(f"{'='*60}")
    print(f"MULTIMODAL EXTRACTION PIPELINE (BATCH): {task_id}")
    print(f"{'='*60}\n")

    # Step 1: Prepare dataset with frames and transcripts
    dataset_path = prepare_dataset(task_id)

    # Step 2: Extract in parallel
    knowledge_items = await extract_knowledge_batch(
        task_id, dataset_path, max_segments=max_segments
    )

    # Step 3: Save
    output_path = save_knowledge_base(knowledge_items, task)

    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"Dataset: {dataset_path}")
    print(f"Knowledge base: {output_path}")
    print(f"Total items: {len(knowledge_items)}")


if __name__ == "__main__":
    # Note: Make sure the task_id exists in static/tasks.json first
    # You can create it using: python -m scripts.prepare_dataset list

    # Example 1: Small test (synchronous)
    # run_pipeline_sync(
    #     task_id="sushi_v1",
    #     max_segments=5
    # )

    # Example 2: Larger batch (parallel)
    # asyncio.run(run_pipeline_batch(
    #     task_id="sushi_v1",
    #     max_segments=50
    # ))

    print("Examples defined. Uncomment to run.")
