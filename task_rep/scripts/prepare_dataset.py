#!/usr/bin/env python3
"""
CLI script to prepare datasets.

Usage:
    # List all tasks
    python -m task_rep.scripts.prepare_dataset list

    # Show task info
    python -m task_rep.scripts.prepare_dataset info sushi_v1

    # Build dataset (OpenAI transcripts by default)
    python -m task_rep.scripts.prepare_dataset build sushi_v1 --fps 1.0

    # Build all datasets at once
    python -m task_rep.scripts.prepare_dataset build-all --fps 1.0

    # Use YouTube transcripts when available
    python -m task_rep.scripts.prepare_dataset build sushi_v1 --transcript-source youtube-or-openai

    # Force rebuild everything
    python -m task_rep.scripts.prepare_dataset build sushi_v1 --force-all
"""

import argparse
import json
from pathlib import Path
from data_prep import TaskManager, build_dataset, VideoDownloader

TASKS_FILE = Path("./static/tasks.json")

VIDEOS_DIR = Path("./static/videos/")

DATASETS_DIR = Path("./static/datasets/")

def cmd_build(args):
    """Build a dataset from task definition."""
    manager = TaskManager(tasks_file=TASKS_FILE)
    task = manager.get_task(args.task_id)
    fps = args.fps
    transcript_source = args.transcript_source
    force_download = args.force_download

    downloader = VideoDownloader(
        videos_dir=VIDEOS_DIR,
    )

    output_path = build_dataset(
        task=task,
        base_dir=DATASETS_DIR,
        downloader=downloader,
        fps=fps,
        transcript_source=transcript_source,
        force_download=force_download,
    )
    print(f"\n✓ Dataset ready: {output_path}")


def cmd_build_all(args):
    """Build datasets for all tasks."""
    manager = TaskManager(tasks_file=TASKS_FILE)
    tasks = manager.list_tasks()

    if not tasks:
        print("No tasks defined yet.")
        return

    print(f"\n{'='*60}")
    print(f"Building {len(tasks)} task(s)")
    print(f"{'='*60}")

    transcript_source = args.transcript_source
    force_download = args.force_download
    fps = args.fps

    success_count = 0
    failed_tasks = []

    for i, task in enumerate(tasks, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(tasks)}] Building task: {task.id}")
        print(f"{'='*60}")

        try:
            downloader = VideoDownloader(videos_dir=VIDEOS_DIR)

            output_path = build_dataset(
                task=task,
                base_dir=DATASETS_DIR,
                downloader=downloader,
                fps=fps,
                transcript_source=transcript_source,
                force_download=force_download,
            )
            print(f"\n✓ Dataset ready: {output_path}")
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


def cmd_status(args):
    """Show download status for one or all tasks."""
    manager = TaskManager(tasks_file=TASKS_FILE)

    if hasattr(args, "task_id") and args.task_id:
        tasks = [manager.get_task(args.task_id)]
    else:
        tasks = manager.list_tasks()

    if not tasks:
        print("No tasks defined yet.")
        return

    totals = {"expected": 0, "in_dataset": 0, "frames": 0,
              "transcript_yt": 0, "transcript_openai": 0}

    print(f"\n{'='*72}")
    print(f"{'Task':<24} {'Exp':>5} {'Ready':>6} {'Frames':>7}  {'Transcript':^20}  {'Miss':>5}")
    print(f"{'':24} {'':5} {'':6} {'':7}  {'YT':>6} {'OAI':>6}  {'':5}")
    print(f"{'='*72}")

    missing_by_task = {}

    for task in tasks:
        expected = len(task.video_ids)
        dataset_path = DATASETS_DIR / f"{task.id}.json"

        if not dataset_path.exists():
            in_dataset = frames = 0
            transcript_yt = transcript_openai = 0
            missing = task.video_ids[:]
        else:
            with open(dataset_path, "r", encoding="utf-8") as f:
                dataset = json.load(f)
            dataset_ids = {v["id"] for v in dataset}
            in_dataset        = len(dataset)
            frames            = sum(1 for v in dataset if v.get("frames"))
            transcript_yt     = sum(1 for v in dataset if v.get("transcript_source") == "youtube")
            transcript_openai = sum(1 for v in dataset if v.get("transcript_source") == "openai")
            missing           = [vid for vid in task.video_ids if vid not in dataset_ids]

        missing_by_task[task.id] = missing

        miss_str = str(len(missing)) if missing else "-"
        frames_pct = f"{frames/expected*100:.0f}%" if expected else "0%"
        print(f"  {task.id:<22} {expected:>5} {in_dataset:>6} {frames:>5}{frames_pct:>3} "
              f" {transcript_yt:>6} {transcript_openai:>6}  {miss_str:>5}")

        totals["expected"]          += expected
        totals["in_dataset"]        += in_dataset
        totals["frames"]            += frames
        totals["transcript_yt"]     += transcript_yt
        totals["transcript_openai"] += transcript_openai

    if len(tasks) > 1:
        print(f"{'─'*72}")
        total_missing = totals["expected"] - totals["frames"] - totals["transcript_yt"] - totals["transcript_openai"]
        miss_str = str(total_missing) if total_missing else "-"
        frames_pct = f"{totals['frames']/totals['expected']*100:.0f}%" if totals["expected"] else "0%"
        print(f"  {'TOTAL':<22} {totals['expected']:>5} {totals['in_dataset']:>6} "
              f"{totals['frames']:>5}{frames_pct:>3} "
              f" {totals['transcript_yt']:>6} {totals['transcript_openai']:>6}  {miss_str:>5}")

    print()


def cmd_list(args):
    """List all tasks."""
    manager = TaskManager(tasks_file=TASKS_FILE)
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
    manager = TaskManager(tasks_file=TASKS_FILE)
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
        choices=["openai", "youtube"],
        default="openai",
        help="Transcript source: 'openai' (default, most reliable) or 'youtube' (prefer YouTube)"
    )
    build_parser.add_argument("--force-download", action="store_true", help="Re-do everything (download, extract frames, get transcript)")

    # Build all command
    build_all_parser = subparsers.add_parser("build-all", help="Build datasets for all tasks")
    build_all_parser.add_argument("--fps", type=float, default=1.0, help="Frames per second (default: 1.0)")
    build_all_parser.add_argument(
        "--transcript-source",
        choices=["openai", "youtube"],
        default="openai",
        help="Transcript source: 'openai' (default, most reliable) or 'youtube' (prefer YouTube)"
    )
    build_all_parser.add_argument("--force-download", action="store_true", help="Re-do everything (download, extract frames, get transcript)")

    # Status command
    status_parser = subparsers.add_parser("status", help="Show download/processing status")
    status_parser.add_argument("task_id", nargs="?", help="Task ID (omit to show all tasks)")

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
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "info":
        cmd_info(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
