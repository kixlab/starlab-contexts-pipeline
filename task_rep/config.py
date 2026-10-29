from pathlib import Path

DATA_DIR = Path("./data/")
STORAGE_DIR = Path("./static/")

class Config:
    TASKS_FILE: Path = DATA_DIR / "tasks.json"
    
    VIDEOS_DIR: Path = STORAGE_DIR / "videos"
    TASK_DATASETS_DIR: Path = STORAGE_DIR / "task-datasets"

    CACHE_DIR: Path = STORAGE_DIR / "cache_2"   # items, units, runs (contextual/cache.py)

    TASKS_PER_DATASET = {
        "test_tasks": [
            "sushi_roll_3",
        ],
        "custom_tasks": [
            "sushi_roll",
            "caramel_apples",
            "milkshake",
            "steak",
            "scrambled_eggs",
            "hydrangea",
            "pumpkin",
            "bathroom_tile",
            "stainless_steel",
            "top_stove",
            "wasp_nest",
            "christmas_tree",
            "boxing",
            "trout",
            "paper_hat"
        ],
        "cross_task_tasks": [
            "tire",
            "floating_shelves",
            "french_toast",
            "irish_coffee"
        ],
        "big_custom_tasks": [
            "sushi_roll",
            "steak",
            "caramel_apples",
            "top_stove"
        ]
    }
