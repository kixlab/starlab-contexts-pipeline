"""
Experiment registry for iterative task schema discovery.

Define, run, save, and compare experiments across three axes:
  1. Datasets   — which task + how many videos to include
  2. Extraction — granularity (ATOMIC/STEP) + clustering threshold + model
  3. Discovery  — algorithm hyperparameters + VLM provider/model

Experiments are stored as JSON files in task_rep/experiments/.
Use load_experiment() / save_experiment() to work with them, or
drop a JSON file into that directory and it will be auto-loaded.

Naming convention for experiment files:
  {task_id}__{item_type}__{variant}.json
  e.g.  sushi_roll_3__step__baseline.json
        sushi_roll_3__atomic__gpt4o.json
        sushi_roll_3__step__high_iter_v2.json

The double-underscore separates logical parts so you can glob by task or type:
  ls experiments/sushi_roll_3__*          # all experiments for a task
  ls experiments/*__step__*               # all step experiments

Usage (library):
    from experiments import EXPERIMENTS, run_experiment, compare_results
    from experiments import load_experiment, save_experiment

    result = run_experiment(EXPERIMENTS["sushi_roll_3__step__baseline"])
    compare_results([result])

    # Save a config you built in code to a file:
    save_experiment(my_config)

    # Load a single file:
    cfg = load_experiment("experiments/sushi_roll_3__step__baseline.json")

Usage (script):
    python -m experiments --list
    python -m experiments --run sushi_roll_3__step__baseline
    python -m experiments --save-config sushi_roll_3__step__baseline

All paths are relative to the task_rep/ directory.
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from config import Config

from algorithm import (
    DiscoveryConfig,
    DiscoveryResult,
    IterativeSchemaDiscovery,
)

from construct import (
    ConstructKnowledgeBase,
    ConstructorConfig,
)

from data_prep import (
    TaskManager,
)

from optimization import (
    count_unique_units,
    compute_tree_depth,
)

from knowledge_items.io import (
    load_task_dataset,
    load_knowledge_items,
    get_extraction_cache_path,
    get_clustering_cache_path,
)
from knowledge_items.schema import KnowledgeItemType, KnowledgeBase

# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Full specification of one experiment run."""

    name: str
    """Short identifier used as a key in EXPERIMENTS and for file naming."""

    task_id: str
    """Task ID to include in the experiment."""

    constructor: ConstructorConfig
    discovery: DiscoveryConfig

    description: str = ""
    """Free-text notes about what this experiment is testing."""

    tags: List[str] = field(default_factory=list)
    """Labels for filtering/grouping (e.g. ['baseline', 'sushi', 'atomic'])."""

    def cache_suffix(self) -> str:
        constructor_suffix = self.constructor.cache_suffix()
        discovery_suffix = self.discovery.cache_suffix()
        return f"{constructor_suffix}_{discovery_suffix}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "task_id": self.task_id,
            "description": self.description,
            "tags": self.tags,
            "constructor": self.constructor.to_dict(),
            "discovery": self.discovery.to_dict(),
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ExperimentConfig":
        return ExperimentConfig(
            name=data["name"],
            task_id=data["task_id"],
            description=data.get("description", ""),
            tags=data.get("tags", []),
            constructor=ConstructorConfig.from_dict(data["constructor"]),
            discovery=DiscoveryConfig.from_dict(data["discovery"]),
        )

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ExperimentSummary:
    """Key metrics for quick cross-experiment comparison."""

    num_units: int
    """Distinct knowledge units in the knowledge base."""

    num_facets: int
    """Facets in the final facet pool."""

    iterations: int
    """How many algorithm iterations ran before convergence."""

    tree_nodes: int
    """Surviving nodes in the refinement tree."""

    tree_depth: int
    """Max path length from root to leaf in the refinement tree."""

    discovery_calls: int
    """Total VLM discovery/revision calls (init + propose + revise)."""


@dataclass
class ExperimentResult:
    """Output of a single experiment run."""

    run_id: str
    """Short unique ID for this run (8-char hex)."""

    timestamp: str
    """ISO 8601 UTC timestamp of when the run started."""

    config: ExperimentConfig
    result: DiscoveryResult
    summary: ExperimentSummary
    elapsed_seconds: float


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def run_experiment(
    config: ExperimentConfig,
    *,
    verbose: bool = True,
) -> ExperimentResult:
    """
    Run a full experiment end-to-end.

    Args:
        config: Full experiment specification.
        verbose: Enable INFO-level logging and progress prints.

    Returns:
        ExperimentResult with the discovery output and run metadata.
    """

    if verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(levelname)s %(name)s: %(message)s",
            force=True,
        )

    t0 = time.monotonic()

    constructor = ConstructKnowledgeBase(
        config=config.constructor,
        task_datasets_dir=Config.TASK_DATASETS_DIR,
        cache_dir=Config.CACHE_DIR,
    )

    discovery = IterativeSchemaDiscovery(
        config=config.discovery
    )

    manager = TaskManager(tasks_file=Config.TASKS_FILE)

    if verbose:
        elapsed = time.monotonic() - t0
        logging.info(f"{elapsed:.2f}s: Prepared constructor and discovery.\nRunning experiment for {config.task_id}.")

    task = manager.get_task(config.task_id)
    if verbose:
        elapsed = time.monotonic() - t0
        logging.info(f"{elapsed:.2f}s: Running experiment for {task.id}.")

    run_id = str(uuid.uuid4())
    cache_dir = constructor.get_cache_dir(task)
    cache_dir.mkdir(parents=True, exist_ok=True)
    log_path = discovery.get_result_path(cache_dir).with_suffix(".log")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n=== {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} run_id={run_id} ===\n")
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(file_handler)

    try:
        result = discovery.run(
            task=task,
            constructor=constructor,
        )
    finally:
        logging.getLogger().removeHandler(file_handler)
        file_handler.close()

    elapsed = time.monotonic() - t0

    num_units = count_unique_units(result.knowledge_base)
    tree_nodes = len(result.tree.nodes) if result.tree else 0
    tree_depth = compute_tree_depth(result.tree) if result.tree else 0

    summary = ExperimentSummary(
        num_units=num_units,
        num_facets=len(result.facets),
        iterations=result.iterations,
        tree_nodes=tree_nodes,
        tree_depth=tree_depth,
        discovery_calls=result.discovery_calls,
    )

    return ExperimentResult(
        run_id=run_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        config=config,
        result=result,
        summary=summary,
        elapsed_seconds=elapsed,
    )


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def compare_results(results: List[ExperimentResult]) -> None:
    """Print a compact side-by-side comparison table."""
    cols = f"{'name':<38} {'units':>6} {'facets':>7} {'t_nodes':>7} {'t_depth':>7} {'d_calls':>7} {'iters':>6} {'secs':>6}"
    print(cols)
    print("-" * len(cols))
    for r in results:
        s = r.summary
        c = r.config
        print(
            f"{c.name:<38} {s.num_units:>6} {s.num_facets:>7} {s.tree_nodes:>7} {s.tree_depth:>7} {s.discovery_calls:>7} {s.iterations:>6} {r.elapsed_seconds:.1f}"
        )


# ---------------------------------------------------------------------------
# Experiment registry
# ---------------------------------------------------------------------------

EXPERIMENTS: Dict[str, ExperimentConfig] = {}
"""Registry mapping experiment name → ExperimentConfig.

Populated automatically from JSON files in the experiments/ directory.
You can also call register_experiment() to add configs at runtime.
"""

EXPERIMENTS_DIR = Path(__file__).parent / "experiments"
"""Directory where experiment JSON files are stored."""


def register_experiment(config: ExperimentConfig) -> ExperimentConfig:
    """Register an ExperimentConfig and return it unchanged."""
    EXPERIMENTS[config.name] = config
    return config


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def load_experiment(path: str | Path) -> ExperimentConfig:
    """Load an ExperimentConfig from a JSON file."""
    data = json.loads(Path(path).read_text())
    return ExperimentConfig.from_dict(data)


def save_experiment(config: ExperimentConfig, path: str | Path | None = None) -> Path:
    """Save an ExperimentConfig to a JSON file.

    If path is None, saves to EXPERIMENTS_DIR/{config.name}.json.
    Returns the path written.
    """
    if path is None:
        path = EXPERIMENTS_DIR / f"{config.name}.json"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config.to_dict(), indent=2))
    return path


def load_experiments_dir(directory: str | Path | None = None) -> Dict[str, ExperimentConfig]:
    """Load all experiment JSON files from a directory.

    Returns a dict mapping experiment name → ExperimentConfig.
    Files are loaded in sorted order; duplicate names (same 'name' field)
    from different files raise a ValueError.
    """
    if directory is None:
        directory = EXPERIMENTS_DIR
    directory = Path(directory)
    result: Dict[str, ExperimentConfig] = {}
    if not directory.exists():
        return result
    for f in sorted(directory.glob("*.json")):
        config = load_experiment(f)
        if config.name in result:
            raise ValueError(
                f"Duplicate experiment name '{config.name}' found in {f}. "
                f"Each experiment file must have a unique 'name' field."
            )
        result[config.name] = config
    return result


# Auto-load experiments from the experiments/ directory.
EXPERIMENTS.update(load_experiments_dir())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _list_experiments() -> None:
    print(f"\n{'name':<40} {'file':<35} {'tags'}")
    print("-" * 90)
    for name, cfg in EXPERIMENTS.items():
        tag_str = ", ".join(cfg.tags) if cfg.tags else "—"
        file_hint = f"experiments/{name}.json"
        print(f"{name:<40} {file_hint:<35} [{tag_str}]")
        if cfg.description:
            print(f"  {cfg.description}")
    print()


def _dataset_info(name: str) -> None:
    if name not in EXPERIMENTS:
        print(f"Unknown experiment '{name}'. Available: {list(EXPERIMENTS)}")
        return

    config = EXPERIMENTS[name]
    manager = TaskManager(tasks_file=Config.TASKS_FILE)
    task = manager.get_task(config.task_id)

    # --- Dataset info ---
    dataset_path = task.get_dataset_path(Config.TASK_DATASETS_DIR)
    videos = load_task_dataset(dataset_path)
    if config.constructor.max_videos is not None:
        videos = videos[: config.constructor.max_videos]

    num_videos = len(videos)
    avg_duration = sum(v.get_duration() for v in videos) / num_videos if num_videos > 0 else 0.0

    print(f"\n=== Dataset Info: {name} ===")
    print(f"Task ID:           {config.task_id}")
    print(f"Videos:            {num_videos}")
    print(f"Avg duration:      {avg_duration:.1f}s")

    # --- Knowledge info ---
    cache_dir = task.get_cache_dir(Config.CACHE_DIR, config.constructor.cache_suffix())
    item_type = config.discovery.item_scope.item_type

    clustering_path = get_clustering_cache_path(cache_dir, item_type, config.constructor.cluster_config)
    extraction_path = get_extraction_cache_path(cache_dir, item_type)

    if clustering_path.exists():
        kb = KnowledgeBase(items=load_knowledge_items(clustering_path))
    elif extraction_path.exists():
        kb = KnowledgeBase(items=load_knowledge_items(extraction_path))
    else:
        kb = None

    if kb is not None:
        by_video = kb.by_video()
        num_items = len(kb)
        num_videos_with_items = len(by_video)
        avg_items = num_items / num_videos_with_items if num_videos_with_items > 0 else 0.0
        print(f"Avg {item_type.value}s/video: {avg_items:.1f}  (total: {num_items} across {num_videos_with_items} videos)")
    else:
        print("No knowledge info")

    # --- Algorithm info ---
    result_path = cache_dir / f"run_result_{config.discovery.cache_suffix()}.json"

    if result_path.exists():
        result = DiscoveryResult.from_dict(json.loads(result_path.read_text()))

        num_units = count_unique_units(result.knowledge_base)
        tree_nodes = len(result.tree.nodes) if result.tree else 0
        tree_depth = compute_tree_depth(result.tree) if result.tree else 0

        summary = ExperimentSummary(
            num_units=num_units,
            num_facets=len(result.facets),
            iterations=result.iterations,
            tree_nodes=tree_nodes,
            tree_depth=tree_depth,
            discovery_calls=result.discovery_calls,
        )

        print(f"\n--- Experiment Summary ---")
        print(f"Units:             {summary.num_units}")
        print(f"Facets:            {summary.num_facets}")
        print(f"Iterations:        {summary.iterations}")
        print(f"Tree nodes:        {summary.tree_nodes}")
        print(f"Tree depth:        {summary.tree_depth}")
        print(f"Discovery calls:   {summary.discovery_calls}")
    else:
        print("No algorithm info")

    print()


def _budget_analysis(name: str) -> None:
    """Show how tree quality changes as max_facets increases."""
    if name not in EXPERIMENTS:
        print(f"Unknown experiment '{name}'. Available: {list(EXPERIMENTS)}")
        return

    config = EXPERIMENTS[name]
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        force=True,
    )

    constructor = ConstructKnowledgeBase(
        config=config.constructor,
        task_datasets_dir=Config.TASK_DATASETS_DIR,
        cache_dir=Config.CACHE_DIR,
    )
    discovery = IterativeSchemaDiscovery(config=config.discovery)
    manager = TaskManager(tasks_file=Config.TASKS_FILE)
    task = manager.get_task(config.task_id)

    rows = discovery.analyze_budget(task=task, constructor=constructor)
    if not rows:
        print(f"No facet bank found for '{name}' (mode: {config.discovery.selection_mode}). Run the experiment first.")
        return

    # Print progression table
    header = f"{'budget':>7} {'t_nodes':>8} {'t_ufacets':>10} {'pool':>50}"
    print(f"\n=== Budget Analysis: {name} (target_facets={config.discovery.target_facets}) ===\n")
    print(header)
    print("-" * len(header))

    prev_nodes = None
    for row in rows:
        marker = ""
        if prev_nodes is not None and row["tree_nodes"] != prev_nodes:
            marker = " *"
        prev_nodes = row["tree_nodes"]
        pool_str = ", ".join(row["pool_ids"])
        if len(pool_str) > 50:
            pool_str = pool_str[:47] + "..."
        print(f"{row['budget']:>7} {row['tree_nodes']:>8} {row['unique_facets_in_tree']:>10} {pool_str:>50}{marker}")

    print(f"\n* = tree_nodes changed from previous row")
    print(f"Bank size: {rows[-1]['budget']}  |  Final tree: {rows[-1]['tree_nodes']} nodes\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run task schema discovery experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--list", action="store_true", help="List all registered experiments.")
    parser.add_argument("--run", metavar="NAME", help="Run an experiment by name.")
    parser.add_argument(
        "--save-config",
        metavar="NAME",
        help="Save an experiment's config to experiments/<NAME>.json.",
    )
    parser.add_argument(
        "--dataset-info",
        metavar="NAME",
        help="Show dataset statistics and experiment summary for NAME.",
    )
    parser.add_argument(
        "--budget-analysis",
        metavar="NAME",
        help="Show how tree quality changes as max_facets increases (uses facet bank).",
    )
    args = parser.parse_args()

    if args.list:
        _list_experiments()
        return

    if args.save_config:
        name = args.save_config
        if name not in EXPERIMENTS:
            print(f"Unknown experiment '{name}'. Available: {list(EXPERIMENTS)}")
            return
        path = save_experiment(EXPERIMENTS[name])
        print(f"Saved to {path}")
        return

    if args.dataset_info:
        _dataset_info(args.dataset_info)
        return

    if args.budget_analysis:
        _budget_analysis(args.budget_analysis)
        return

    if args.run:
        if args.run not in EXPERIMENTS:
            names = list(EXPERIMENTS)
            print(f"Unknown experiment '{args.run}'.")
            print(f"Available: {names}")
            return
        result = run_experiment(EXPERIMENTS[args.run], verbose=True)
        compare_results([result])
        return

    parser.print_help()


if __name__ == "__main__":
    main()
