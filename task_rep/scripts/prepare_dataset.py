#!/usr/bin/env python3
"""
CLI script to prepare datasets.

Usage:
    # List all tasks
    python -m task_rep.scripts.prepare_dataset list

    # Show task info
    python -m task_rep.scripts.prepare_dataset info sushi_v1

    # Build dataset (Whisper transcripts by default)
    python -m task_rep.scripts.prepare_dataset build sushi_v1 --fps 1.0

    # Build all datasets at once
    python -m task_rep.scripts.prepare_dataset build-all --fps 1.0

    # Use YouTube transcripts when available
    python -m task_rep.scripts.prepare_dataset build sushi_v1 --transcript-source youtube-or-whisper

    # Force rebuild everything
    python -m task_rep.scripts.prepare_dataset build sushi_v1 --force-all
"""

import argparse

from data_prep import TaskManager, build_dataset


def cmd_build(args):
    """Build a dataset from task definition."""
    output_path = build_dataset(
        task_id=args.task_id,
        fps=args.fps,
        transcript_source=args.transcript_source,
        force_download=args.force_download or args.force_all,
        force_extract=args.force_extract or args.force_all,
        force_transcripts=args.force_transcripts or args.force_all,
    )
    print(f"\n✓ Dataset ready: {output_path}")


def cmd_build_all(args):
    """Build datasets for all tasks."""
    manager = TaskManager()
    tasks = manager.list_tasks()

    if not tasks:
        print("No tasks defined yet.")
        return

    print(f"\n{'='*60}")
    print(f"Building {len(tasks)} task(s)")
    print(f"{'='*60}")

    success_count = 0
    failed_tasks = []

    for i, task in enumerate(tasks, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(tasks)}] Building task: {task.id}")
        print(f"{'='*60}")

        try:
            output_path = build_dataset(
                task_id=task.id,
                fps=args.fps,
                transcript_source=args.transcript_source,
                force_download=args.force_download or args.force_all,
                force_extract=args.force_extract or args.force_all,
                force_transcripts=args.force_transcripts or args.force_all,
            )
            print(f"\n✓ Task {task.id} completed: {output_path}")
            success_count += 1
        except Exception as e:
            print(f"\n✗ Task {task.id} failed: {e}")
            failed_tasks.append((task.id, str(e)))

    # Summary
    print(f"\n{'='*60}")
    print(f"Build Summary")
    print(f"{'='*60}")
    print(f"  Total tasks: {len(tasks)}")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {len(failed_tasks)}")

    if failed_tasks:
        print(f"\nFailed tasks:")
        for task_id, error in failed_tasks:
            print(f"  - {task_id}: {error}")


def cmd_list(args):
    """List all tasks."""
    manager = TaskManager()
    tasks = manager.list_tasks()

    if not tasks:
        print("No tasks defined yet.")
        print("\nCreate tasks.json with this format:")
        print("""
{
  "task_id": {
    "name": "Task Name",
    "video_ids": ["video_id1", "video_id2", ...]
  }
}
""")
        return

    print(f"\nDefined Tasks ({len(tasks)}):\n")
    for task in tasks:
        print(f"  {task.id}")
        print(f"    Name: {task.name}")
        print(f"    Videos: {len(task.video_ids)}")
        print()


def cmd_info(args):
    """Show task information."""
    manager = TaskManager()
    task = manager.get_task(args.task_id)

    print(f"\nTask: {task.id}")
    print(f"  Name: {task.name}")
    print(f"  Videos: {len(task.video_ids)}")
    print(f"\n  Video IDs:")
    for video_id in task.video_ids[:10]:
        print(f"    - {video_id}")
    if len(task.video_ids) > 10:
        print(f"    ... and {len(task.video_ids) - 10} more")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare datasets from tasks.json"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Build command
    build_parser = subparsers.add_parser("build", help="Build dataset from task")
    build_parser.add_argument("task_id", help="Task ID from tasks.json")
    build_parser.add_argument("--fps", type=float, default=1.0, help="Frames per second (default: 1.0)")
    build_parser.add_argument(
        "--transcript-source",
        choices=["whisper", "youtube-or-whisper"],
        default="whisper",
        help="Transcript source: 'whisper' (default, most reliable) or 'youtube-or-whisper' (prefer original)"
    )
    build_parser.add_argument("--force-download", action="store_true", help="Re-download videos")
    build_parser.add_argument("--force-extract", action="store_true", help="Re-extract frames")
    build_parser.add_argument("--force-transcripts", action="store_true", help="Re-fetch transcripts")
    build_parser.add_argument("--force-all", action="store_true", help="Force rebuild everything")

    # Build all command
    build_all_parser = subparsers.add_parser("build-all", help="Build datasets for all tasks")
    build_all_parser.add_argument("--fps", type=float, default=1.0, help="Frames per second (default: 1.0)")
    build_all_parser.add_argument(
        "--transcript-source",
        choices=["whisper", "youtube-or-whisper"],
        default="whisper",
        help="Transcript source: 'whisper' (default, most reliable) or 'youtube-or-whisper' (prefer original)"
    )
    build_all_parser.add_argument("--force-download", action="store_true", help="Re-download videos")
    build_all_parser.add_argument("--force-extract", action="store_true", help="Re-extract frames")
    build_all_parser.add_argument("--force-transcripts", action="store_true", help="Re-fetch transcripts")
    build_all_parser.add_argument("--force-all", action="store_true", help="Force rebuild everything")

    # List command
    list_parser = subparsers.add_parser("list", help="List all tasks")

    # Info command
    info_parser = subparsers.add_parser("info", help="Show task information")
    info_parser.add_argument("task_id", help="Task ID")

    args = parser.parse_args()

    if args.command == "build":
        cmd_build(args)
    elif args.command == "build-all":
        cmd_build_all(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "info":
        cmd_info(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
