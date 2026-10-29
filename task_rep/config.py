"""Repository-relative defaults, optionally overridden with CIM_STORAGE_DIR."""
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
STORAGE_DIR = Path(os.environ.get("CIM_STORAGE_DIR", str(ROOT_DIR / "static"))).expanduser().resolve()

class Config:
    TASKS_FILE = Path(os.environ.get("CIM_TASKS_FILE", str(DATA_DIR / "tasks.json"))).expanduser().resolve()
    VIDEOS_DIR = STORAGE_DIR / "videos"
    TASK_DATASETS_DIR = STORAGE_DIR / "task-datasets"
    CACHE_DIR = STORAGE_DIR / "cache_2"
    TASKS_PER_DATASET = {"example_tasks": ["paper_folding_demo"]}
