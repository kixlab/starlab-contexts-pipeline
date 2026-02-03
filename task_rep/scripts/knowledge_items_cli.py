#!/usr/bin/env python
"""
CLI for working with knowledge items.

Usage:
    # Extract and cluster knowledge items from a task
    python -m task_rep.scripts.knowledge_items_cli extract sushi_roll_3 --granularity atomic

    # Inspect cached knowledge items
    python -m task_rep.scripts.knowledge_items_cli inspect sushi_roll_3 --granularity atomic
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from task_rep.knowledge_items import (
    extract_and_cluster_knowledge_items,
    load_knowledge_items,
    get_extraction_cache_path,
    get_clustering_cache_path,
    ClusterConfig,
    KnowledgeItemType,
    Video,
    VideoTranscriptSegment,
    VideoFrame,
)


def load_dataset(dataset_path: Path) -> list[Video]:
    """Load videos from dataset JSON."""
    with open(dataset_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    videos = []
    for entry in raw:
        transcript_segments = [
            VideoTranscriptSegment(
                text=seg["text"],
                start=float(seg["start"]),
                end=float(seg["end"]),
            )
            for seg in entry.get("transcript", [])
        ]

        frames = []
        if "frames" in entry:
            frames = [
                VideoFrame(
                    path=f["path"],
                    time=float(f["time"]),
                    caption=f.get("caption"),
                )
                for f in entry["frames"]
            ]

        video = Video(
            id=entry["id"],
            url=entry["url"],
            title=entry["title"],
            original_title=entry.get("original_title", entry["title"]),
            content=entry.get("content", "") or "",
            transcript=transcript_segments,
            frames=frames,
        )
        videos.append(video)

    return videos


def cmd_extract(args):
    """Extract and cluster knowledge items from a task."""
    dataset_path = Path("static/datasets") / f"{args.task_id}.json"

    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        print(f"   Run: python -m task_rep.scripts.prepare_dataset build {args.task_id}")
        return 1

    print(f"Loading dataset: {dataset_path}")
    videos = load_dataset(dataset_path)
    print(f"✓ Loaded {len(videos)} videos")

    # Parse granularity
    granularity = KnowledgeItemType(args.granularity)

    # Create cluster config if threshold provided
    cluster_config = None
    if args.cluster_threshold is not None:
        cluster_config = ClusterConfig(
            similarity_threshold=args.cluster_threshold,
            embedding_model=args.embedding_model,
        )

    # Extract and cluster
    print(f"\nExtracting {granularity.value} knowledge items...")
    if cluster_config:
        print(f"  Clustering: threshold={cluster_config.similarity_threshold}, model={cluster_config.embedding_model}")

    knowledge_base = extract_and_cluster_knowledge_items(
        task_id=args.task_id,
        granularity=granularity,
        videos=videos,
        base_dir=Path("static/cache"),
        provider=args.provider,
        model=args.model,
        cluster_config=cluster_config,
        use_cache=not args.no_cache,
    )

    print(f"\n✓ Extracted {len(knowledge_base.items)} knowledge items")

    if cluster_config:
        unique_units = len(set(item.unit_id for item in knowledge_base.items if item.unit_id))
        print(f"✓ Clustered into {unique_units} unique units")

    # Show sample
    if args.show_sample and knowledge_base.items:
        print(f"\n--- Sample items ---")
        for i, item in enumerate(knowledge_base.items[:args.show_sample]):
            print(f"\n{i+1}. [{item.type.value}] {item.subtype if item.subtype else 'N/A'}")
            print(f"   Unit: {item.unit_id}")
            print(f"   Text: {item.text[:100]}...")

    return 0


def cmd_inspect(args):
    """Inspect cached knowledge items."""
    granularity = KnowledgeItemType(args.granularity)
    base_dir = Path("static/cache")

    # Check extraction cache
    extraction_cache = get_extraction_cache_path(base_dir, args.task_id, granularity)
    clustering_cache = None

    if args.cluster_threshold is not None:
        config = ClusterConfig(
            similarity_threshold=args.cluster_threshold,
            embedding_model=args.embedding_model,
        )
        clustering_cache = get_clustering_cache_path(base_dir, args.task_id, granularity, config)

    # Try loading
    items = None
    source = None

    if clustering_cache and clustering_cache.exists():
        print(f"Loading from clustering cache: {clustering_cache}")
        items = load_knowledge_items(clustering_cache)
        source = "clustered"
    elif extraction_cache.exists():
        print(f"Loading from extraction cache: {extraction_cache}")
        items = load_knowledge_items(extraction_cache)
        source = "extracted"
    else:
        print(f"❌ No cached data found for task '{args.task_id}' with granularity '{granularity.value}'")
        print(f"   Extraction cache: {extraction_cache}")
        if clustering_cache:
            print(f"   Clustering cache: {clustering_cache}")
        return 1

    # Print stats
    print(f"\n=== Statistics ({source}) ===")
    print(f"Total items: {len(items)}")

    # Type distribution
    type_counts = {}
    for item in items:
        type_counts[item.type.value] = type_counts.get(item.type.value, 0) + 1
    print(f"\nType distribution:")
    for type_name, count in sorted(type_counts.items()):
        print(f"  {type_name}: {count}")

    # Subtype distribution (if atomic)
    if granularity == KnowledgeItemType.ATOMIC:
        subtype_counts = {}
        for item in items:
            subtype = str(item.subtype) if item.subtype else "None"
            subtype_counts[subtype] = subtype_counts.get(subtype, 0) + 1
        print(f"\nSubtype distribution (top 10):")
        for subtype, count in sorted(subtype_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {subtype}: {count}")

    # Unit stats (if clustered)
    if source == "clustered":
        unique_units = len(set(item.unit_id for item in items if item.unit_id))
        print(f"\nClustering:")
        print(f"  Unique units: {unique_units}")
        print(f"  Items per unit (avg): {len(items) / unique_units:.2f}")

    # Video distribution
    video_counts = {}
    for item in items:
        video_id = item.source.video_id
        video_counts[video_id] = video_counts.get(video_id, 0) + 1
    print(f"\nPer-video distribution:")
    for video_id, count in sorted(video_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {video_id}: {count} items")

    # Show samples
    if args.show_sample:
        print(f"\n=== Sample items (first {args.show_sample}) ===")
        for i, item in enumerate(items[:args.show_sample]):
            print(f"\n{i+1}. ID: {item.id}")
            print(f"   Type: {item.type.value}")
            print(f"   Subtype: {item.subtype}")
            print(f"   Unit: {item.unit_id}")
            print(f"   Source: {item.source.video_id}")
            print(f"   Text: {item.text[:150]}...")

    return 0


def main():
    parser = argparse.ArgumentParser(description="CLI for working with knowledge items")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Extract command
    extract_parser = subparsers.add_parser("extract", help="Extract and cluster knowledge items")
    extract_parser.add_argument("task_id", help="Task ID (e.g., sushi_roll_3)")
    extract_parser.add_argument(
        "--granularity",
        choices=["atomic", "step", "tutorial"],
        default="atomic",
        help="Knowledge item granularity",
    )
    extract_parser.add_argument(
        "--cluster-threshold",
        type=float,
        help="Similarity threshold for clustering (0.0-1.0, default: no clustering)",
    )
    extract_parser.add_argument(
        "--embedding-model",
        default="text-embedding-3-small",
        help="Embedding model for clustering",
    )
    extract_parser.add_argument(
        "--provider",
        default="openai",
        help="LLM provider (openai, anthropic)",
    )
    extract_parser.add_argument(
        "--model",
        help="LLM model (default: provider-specific default)",
    )
    extract_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore cache and always extract fresh",
    )
    extract_parser.add_argument(
        "--show-sample",
        type=int,
        default=0,
        help="Show N sample items after extraction",
    )

    # Inspect command
    inspect_parser = subparsers.add_parser("inspect", help="Inspect cached knowledge items")
    inspect_parser.add_argument("task_id", help="Task ID")
    inspect_parser.add_argument(
        "--granularity",
        choices=["atomic", "step", "tutorial"],
        default="atomic",
        help="Knowledge item granularity",
    )
    inspect_parser.add_argument(
        "--cluster-threshold",
        type=float,
        help="Similarity threshold (to load clustered cache)",
    )
    inspect_parser.add_argument(
        "--embedding-model",
        default="text-embedding-3-small",
        help="Embedding model",
    )
    inspect_parser.add_argument(
        "--show-sample",
        type=int,
        default=3,
        help="Show N sample items",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "extract":
        return cmd_extract(args)
    elif args.command == "inspect":
        return cmd_inspect(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
