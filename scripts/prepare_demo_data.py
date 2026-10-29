"""Copy the synthetic transcript dataset into the configured local data folder."""
from pathlib import Path
import os
import shutil

root = Path(__file__).resolve().parents[1]
storage = Path(os.environ.get("CIM_STORAGE_DIR", str(root / "task_rep" / "static"))).expanduser().resolve()
destination = storage / "task-datasets" / "paper_folding_demo.json"
destination.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(root / "examples" / "demo-dataset.json", destination)
print(destination)
