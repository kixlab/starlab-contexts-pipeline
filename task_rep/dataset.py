#!/usr/bin/env python3
"""
CLI to prepare task datasets (run from task_rep/; see data_prep/README.md).

Usage:
    # List all tasks
    python -m dataset list

    # Show task info
    python -m dataset info sushi_v1

    # Build dataset (OpenAI transcripts by default)
    python -m dataset build sushi_v1 --fps 1.0

    # Build all datasets at once
    python -m dataset build-all --fps 1.0

    # Build only specific tasks
    python -m dataset build-some makeup caramel_apples --fps 1.0

    # Use YouTube transcripts when available
    python -m dataset build sushi_v1 --transcript-source youtube-or-openai

    # Force rebuild everything
    python -m dataset build sushi_v1 --force-all
"""

import argparse
import json
import shutil
from pathlib import Path
from config import Config
from data_prep import (
    TaskManager,
    build_task_dataset,
    VideoDownloader,
)
from data_prep.narration_checker import DEFAULT_MIN_WORD_COUNT, DEFAULT_MIN_WORDS_PER_MINUTE

def cmd_build(args):
    """Build a dataset from task definition."""
    manager = TaskManager(tasks_file=Config.TASKS_FILE)
    task = manager.get_task(args.task_id)
    fps = args.fps
    transcript_source = args.transcript_source
    force_download = args.force_download

    downloader = VideoDownloader(videos_dir=Config.VIDEOS_DIR)

    output_path = build_task_dataset(
        task=task,
        downloader=downloader,
        dataset_dir=Config.TASK_DATASETS_DIR,
        fps=fps,
        transcript_source=transcript_source,
        force_download=force_download,
        narration_check=not args.no_narration_check,
        min_word_count=args.min_word_count,
        min_words_per_minute=args.min_words_per_minute,
    )
    print(f"\n✓ Dataset ready: {output_path}")


def cmd_build_all(args):
    """Build datasets for all tasks."""
    manager = TaskManager(tasks_file=Config.TASKS_FILE)
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
    narration_check = not args.no_narration_check
    min_word_count = args.min_word_count
    min_words_per_minute = args.min_words_per_minute

    success_count = 0
    failed_tasks = []

    for i, task in enumerate(tasks, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(tasks)}] Building task: {task.id}")
        print(f"{'='*60}")

        try:
            downloader = VideoDownloader(videos_dir=Config.VIDEOS_DIR)

            output_path = build_task_dataset(
                task=task,
                downloader=downloader,
                dataset_dir=Config.TASK_DATASETS_DIR,
                fps=fps,
                transcript_source=transcript_source,
                force_download=force_download,
                narration_check=narration_check,
                min_word_count=min_word_count,
                min_words_per_minute=min_words_per_minute,
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


def cmd_build_some(args):
    """Build datasets for a specified subset of tasks."""
    manager = TaskManager(tasks_file=Config.TASKS_FILE)
    tasks = [manager.get_task(task_id) for task_id in args.task_ids]

    print(f"\n{'='*60}")
    print(f"Building {len(tasks)} task(s)")
    print(f"{'='*60}")

    transcript_source = args.transcript_source
    force_download = args.force_download
    fps = args.fps
    narration_check = not args.no_narration_check
    min_word_count = args.min_word_count
    min_words_per_minute = args.min_words_per_minute

    success_count = 0
    failed_tasks = []

    for i, task in enumerate(tasks, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(tasks)}] Building task: {task.id}")
        print(f"{'='*60}")

        try:
            downloader = VideoDownloader(videos_dir=Config.VIDEOS_DIR)

            output_path = build_task_dataset(
                task=task,
                downloader=downloader,
                dataset_dir=Config.TASK_DATASETS_DIR,
                fps=fps,
                transcript_source=transcript_source,
                force_download=force_download,
                narration_check=narration_check,
                min_word_count=min_word_count,
                min_words_per_minute=min_words_per_minute,
            )
            print(f"\n✓ Dataset ready: {output_path}")
            success_count += 1
        except Exception as e:
            print(f"\n✗ Task {task.id} failed: {e}")
            failed_tasks.append((task.id, str(e)))

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
    manager = TaskManager(tasks_file=Config.TASKS_FILE)

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
        dataset_path = task.get_dataset_path(Config.TASK_DATASETS_DIR)

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
    manager = TaskManager(tasks_file=Config.TASKS_FILE)
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
    manager = TaskManager(tasks_file=Config.TASKS_FILE)
    task = manager.get_task(args.task_id)

    print(f"\nTask: {task.id}")
    print(f"  Name: {task.name}")
    print(f"  Videos: {len(task.video_ids)}")
    print(f"\n  Video IDs:")
    for video_id in task.video_ids[:10]:
        print(f"    - {video_id}")
    if len(task.video_ids) > 10:
        print(f"    ... and {len(task.video_ids) - 10} more")


def cmd_collect(args):
    """Copy in-use video directories to a new location."""
    manager = TaskManager(tasks_file=Config.TASKS_FILE)

    if args.task_ids:
        tasks = [manager.get_task(tid) for tid in args.task_ids]
    else:
        tasks = manager.list_tasks()

    used_ids = set()
    for task in tasks:
        used_ids.update(task.video_ids)

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    src_dir = Path(args.src)
    all_dirs = set(p.name for p in src_dir.iterdir() if p.is_dir()) if src_dir.exists() else set()

    copied = 0
    skipped = 0
    missing = 0

    print(f"\nCollecting {len(used_ids)} in-use videos to {dest}")
    print(f"Source: {src_dir}")
    print(f"Total video dirs on disk: {len(all_dirs)}")
    print()

    for video_id in sorted(used_ids):
        src = src_dir / video_id
        dst = dest / video_id
        if not src.exists():
            print(f"  MISSING  {video_id}")
            missing += 1
            continue
        if dst.exists():
            print(f"  EXISTS   {video_id}")
            skipped += 1
            continue
        shutil.copytree(src, dst)
        print(f"  COPIED   {video_id}")
        copied += 1

    unused = len(all_dirs - used_ids)
    print(f"\n{'='*40}")
    print(f"  Copied:  {copied}")
    print(f"  Existed: {skipped}")
    print(f"  Missing: {missing}")
    print(f"  Unused videos (not copied): {unused}")
    print(f"{'='*40}")


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
    build_parser.add_argument("--no-narration-check", action="store_true", help="Disable narration quality filter")
    build_parser.add_argument("--min-word-count", type=int, default=DEFAULT_MIN_WORD_COUNT, help=f"Minimum word count to pass narration check (default: {DEFAULT_MIN_WORD_COUNT})")
    build_parser.add_argument("--min-words-per-minute", type=float, default=DEFAULT_MIN_WORDS_PER_MINUTE, help=f"Minimum words/min to pass narration check (default: {DEFAULT_MIN_WORDS_PER_MINUTE})")

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
    build_all_parser.add_argument("--no-narration-check", action="store_true", help="Disable narration quality filter")
    build_all_parser.add_argument("--min-word-count", type=int, default=DEFAULT_MIN_WORD_COUNT, help=f"Minimum word count to pass narration check (default: {DEFAULT_MIN_WORD_COUNT})")
    build_all_parser.add_argument("--min-words-per-minute", type=float, default=DEFAULT_MIN_WORDS_PER_MINUTE, help=f"Minimum words/min to pass narration check (default: {DEFAULT_MIN_WORDS_PER_MINUTE})")

    # Build some command
    build_some_parser = subparsers.add_parser("build-some", help="Build datasets for specified tasks")
    build_some_parser.add_argument("task_ids", nargs="+", help="Task IDs to build")
    build_some_parser.add_argument("--fps", type=float, default=1.0, help="Frames per second (default: 1.0)")
    build_some_parser.add_argument(
        "--transcript-source",
        choices=["openai", "youtube"],
        default="openai",
        help="Transcript source: 'openai' (default, most reliable) or 'youtube' (prefer YouTube)"
    )
    build_some_parser.add_argument("--force-download", action="store_true", help="Re-do everything (download, extract frames, get transcript)")
    build_some_parser.add_argument("--no-narration-check", action="store_true", help="Disable narration quality filter")
    build_some_parser.add_argument("--min-word-count", type=int, default=DEFAULT_MIN_WORD_COUNT, help=f"Minimum word count to pass narration check (default: {DEFAULT_MIN_WORD_COUNT})")
    build_some_parser.add_argument("--min-words-per-minute", type=float, default=DEFAULT_MIN_WORDS_PER_MINUTE, help=f"Minimum words/min to pass narration check (default: {DEFAULT_MIN_WORDS_PER_MINUTE})")

    # Collect command
    collect_parser = subparsers.add_parser("collect", help="Copy in-use videos to a new directory")
    collect_parser.add_argument("--src", default=str(Config.VIDEOS_DIR), help="Source videos directory (default: static/videos)")
    collect_parser.add_argument("--dest", default=str(Config.VIDEOS_DIR.parent / "videos-collected"), help="Destination directory (default: static/videos-collected)")
    collect_parser.add_argument("--task-ids", nargs="+", help="Only collect videos for these tasks (default: all)")

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
    elif args.command == "build-some":
        cmd_build_some(args)
    elif args.command == "collect":
        cmd_collect(args)
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
