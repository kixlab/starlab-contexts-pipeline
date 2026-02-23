"""
Simple task management with tasks.json.

tasks.json format:
{
  "task_id": {
    "name": "Task name",
    "video_ids": ["video_id1", "video_id2", ...]
  }
}
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

@dataclass
class Task:
    """A task definition."""

    id: str
    name: str
    video_ids: List[str]
    metadata: Dict[str, Any] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "video_ids": self.video_ids,
            "metadata": self.metadata or {},
        }

    @staticmethod
    def from_dict(task_id: str, data: dict) -> Task:
        return Task(
            id=task_id,
            name=data["name"],
            video_ids=data["video_ids"],
            metadata=data.get("metadata", {}),
        )

    def get_dataset_path(self, dataset_dir: str | Path) -> Path:
        return Path(dataset_dir) / f"{self.id}.json"

    def get_cache_dir(self, cache_dir: str | Path, suffix: str = "") -> Path:
        return Path(cache_dir) / f"{self.id}{f'_{suffix}' if suffix else ''}"

class TaskManager:
    """Manages tasks.json."""

    def __init__(self, tasks_file: str | Path):
        self.tasks_file = Path(tasks_file)
        self.tasks_file.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self):
        """Load tasks from file."""
        if self.tasks_file.exists():
            with open(self.tasks_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.tasks = {
                    task_id: Task.from_dict(task_id, task_data)
                    for task_id, task_data in data.items()
                }
        else:
            self.tasks = {}

    def _save(self):
        """Save tasks to file."""
        data = {task_id: task.to_dict() for task_id, task in self.tasks.items()}
        with open(self.tasks_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def create_task(
        self, task_id: str, name: str, video_ids: List[str], metadata: Optional[Dict] = None
    ) -> Task:
        """Create a new task."""
        if task_id in self.tasks:
            raise ValueError(f"Task '{task_id}' already exists")

        task = Task(id=task_id, name=name, video_ids=video_ids, metadata=metadata or {})
        self.tasks[task_id] = task
        self._save()
        return task

    def get_task(self, task_id: str) -> Task:
        """Get a task by ID."""
        if task_id not in self.tasks:
            raise ValueError(f"Task '{task_id}' not found")
        return self.tasks[task_id]

    def list_tasks(self, task_ids: Optional[List[str]] = None) -> List[Task]:
        """List all tasks."""
        if task_ids is None:
            return list(self.tasks.values())
        else:
            return [self.tasks[task_id] for task_id in task_ids]

    def update_task(self, task_id: str, **kwargs) -> Task:
        """Update a task."""
        task = self.get_task(task_id)

        if "name" in kwargs:
            task.name = kwargs["name"]
        if "video_ids" in kwargs:
            task.video_ids = kwargs["video_ids"]
        if "metadata" in kwargs:
            task.metadata.update(kwargs["metadata"])

        self._save()
        return task

    def delete_task(self, task_id: str):
        """Delete a task."""
        if task_id not in self.tasks:
            raise ValueError(f"Task '{task_id}' not found")
        del self.tasks[task_id]
        self._save()
