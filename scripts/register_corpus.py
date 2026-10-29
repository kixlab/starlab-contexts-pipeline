"""Register timed transcripts and a small local experiment; makes no model calls."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def prepare_corpus(data: list) -> list:
    if not isinstance(data, list) or not data:
        raise ValueError("Expected a nonempty array of tutorials with timed transcripts")
    result, ids = [], set()
    for video in data:
        video_id = video.get("id")
        if not isinstance(video_id, str) or not video_id.strip() or video_id in ids:
            raise ValueError("Tutorial IDs must be nonempty, unique strings")
        ids.add(video_id)
        transcript = video.get("transcript")
        if not isinstance(transcript, list) or not transcript:
            raise ValueError(f"Tutorial {video_id} needs a nonempty transcript")
        segments = []
        for segment in transcript:
            start, end, text = segment.get("start"), segment.get("end"), segment.get("text")
            if (not isinstance(text, str) or not text.strip() or
                not all(type(t) in (int, float) and math.isfinite(t) for t in (start, end)) or
                    start < 0 or end < start):
                raise ValueError(f"Tutorial {video_id}: transcript entries need text and seconds with 0 <= start <= end")
            segments.append({"text": text, "start": start, "end": end})
        url = video.get("url", "")
        if not isinstance(url, str):
            raise ValueError(f"Tutorial {video_id}: url must be a string")
        result.append({"id": video_id, "url": url, "title": video.get("title", video_id),
                       "transcript": sorted(segments, key=lambda s: s["start"]), "frames": []})
    return result


def register_corpus(source: Path, task_id: str, name: str, *, root: Path = ROOT, force: bool = False):
    if not re.fullmatch(r"local[-_][a-z0-9][a-z0-9_-]*", task_id):
        raise ValueError("Use a task ID such as local-my-tutorials (local- prefix keeps files ignored)")
    corpus = prepare_corpus(json.loads(source.read_text()))
    dataset_path = root / "task_rep/static/task-datasets" / f"{task_id}.json"
    experiment_path = root / "task_rep/experiments" / f"{task_id}.json"
    tasks_path = root / "task_rep/data/local-tasks.json"
    tasks = json.loads(tasks_path.read_text()) if tasks_path.exists() else json.loads((root / "task_rep/data/tasks.json").read_text())
    if not force and (dataset_path.exists() or experiment_path.exists() or task_id in tasks):
        raise ValueError("This local task already exists; use a new ID or --force to replace its inputs/config")
    tasks[task_id] = {"name": name, "video_ids": [v["id"] for v in corpus]}
    spec = json.loads((root / "task_rep/experiments/paper_folding_demo.json").read_text())
    spec.update(name=task_id, task_id=task_id, description="Local transcript corpus")
    for path, data in [(dataset_path, corpus), (tasks_path, tasks), (experiment_path, spec)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    return dataset_path, experiment_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        dataset, experiment = register_corpus(args.source, args.task_id, args.name, force=args.force)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f"Registered {dataset.relative_to(ROOT)} and {experiment.relative_to(ROOT)}")
    print("Before running: export CIM_TASKS_FILE=task_rep/data/local-tasks.json")
    print(f"Review {experiment.relative_to(ROOT)} before python main.py run {args.task_id} (paid model calls)")


if __name__ == "__main__":
    main()
