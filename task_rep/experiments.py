"""
Experiment registry for iterative task schema discovery.

Define, run, save, and compare experiments across three axes:
  1. Datasets   — which task + how many videos to include
  2. Extraction — granularity (ATOMIC/STEP) + clustering threshold + model
  3. Discovery  — algorithm hyperparameters + VLM provider/model

Usage (library):
    from experiments import EXPERIMENTS, run_experiment, print_summary

    result = run_experiment(EXPERIMENTS["sushi_atomic_baseline"])
    print_summary(result)

Usage (script):
    python -m experiments --list
    python -m experiments --run sushi_atomic_baseline
    python -m experiments --run sushi_atomic_baseline --save-dir results/

All paths are relative to the task_rep/ directory.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
import uuid
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import Config
from utils import ModelConfig

from algorithm import (
    DiscoveryConfig,
    DiscoveryResult,
    IterativeSchemaDiscovery,
)

from knowledge_items import (
    ClusterConfig,
    KnowledgeItemType,
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
)

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

    frontier_size: int
    """Points on the final Pareto frontier."""

    min_discriminativeness: float
    """Minimum D value on the frontier."""
    min_compactness: int
    """Compactness at the minimum D value."""

    target_discriminativeness: float
    """Target D value."""
    target_compactness: int
    """Compactness at the target D value."""

    iterations: int
    """How many algorithm iterations ran before convergence."""


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
    
    result = discovery.run(
        task=task,
        constructor=constructor,
    )
    
    elapsed = time.monotonic() - t0

    num_units = count_unique_units(result.knowledge_base)

    target_discriminativeness = config.discovery.target_d
    target_compactness = 0
    for frontier_point in result.frontier:
        if frontier_point.discriminativeness <= target_discriminativeness:
            target_discriminativeness = frontier_point.discriminativeness
            target_compactness = frontier_point.compactness
            break

    min_discriminativeness = float("inf")
    min_compactness = 0
    if len(result.frontier) > 0:
        min_discriminativeness = result.frontier[-1].discriminativeness
        min_compactness = result.frontier[-1].compactness

    summary = ExperimentSummary(
        num_units=num_units,
        num_facets=len(result.facets),
        frontier_size=len(result.frontier),
        min_discriminativeness=min_discriminativeness,
        min_compactness=min_compactness,
        target_discriminativeness=target_discriminativeness,
        target_compactness=target_compactness,
        iterations=result.iterations,
    )

    return ExperimentResult(
        run_id=str(uuid.uuid4())[:8],
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
    cols = f"{'name':<38} {'units':>6} {'facets':>7} {'frontier':>7} {'min_D':>7} {'min_C':>7} {'target_D':>7} {'target_C':>7} {'iters':>6} {'secs':>6}"
    print(cols)
    print("-" * len(cols))
    for r in results:
        s = r.summary
        c = r.config
        print(
            f"{c.name:<38} {s.num_units:>6} {s.num_facets:>7} {s.frontier_size:>7} {s.min_discriminativeness:>7.4f} {s.min_compactness:>7} {s.target_discriminativeness:>7.4f} {s.target_compactness:>7} {s.iterations:>6} {r.elapsed_seconds:.1f}"
        )


# ---------------------------------------------------------------------------
# Experiment registry
# ---------------------------------------------------------------------------

EXPERIMENTS: Dict[str, ExperimentConfig] = {}
"""Registry mapping experiment name → ExperimentConfig.

Add entries with register_experiment() or by directly assigning:
    EXPERIMENTS["my_exp"] = ExperimentConfig(...)
"""


def register_experiment(config: ExperimentConfig) -> ExperimentConfig:
    """Register an ExperimentConfig and return it unchanged.

    Can be used as a statement (register_experiment(ExperimentConfig(...)))
    or to build the registry incrementally.
    """
    EXPERIMENTS[config.name] = config
    return config


# ---------------------------------------------------------------------------
# Built-in experiments
# ---------------------------------------------------------------------------
# Add your own experiments below.  The three "baseline" entries demonstrate
# the three main axes of variation.
test_task_id = "sushi_roll_3"
item_type = KnowledgeItemType.STEP
register_experiment(ExperimentConfig(
    task_id=test_task_id,

    name=f"test_{test_task_id}_{item_type.value}",
    description=f"test {test_task_id} {item_type.value} experiment v0.1",
    tags=[f"test", f"{test_task_id}", f"{item_type.value}", "v0.1"],
    constructor=ConstructorConfig(
        use_cache=True,
        use_vision=True,
        cluster_config=ClusterConfig(
            similarity_threshold=0.85,
            model_config=ModelConfig(
                category="embedding",
                provider="openai",
                model="text-embedding-3-small",
            ),
        ),
        model_config=ModelConfig(
            category="vlm",
            provider="openai",
            model="gpt-4o-mini",
        ),
    ),
    discovery=DiscoveryConfig(
        use_cache=True,
        item_type=item_type,
        max_iterations=1,
        k_schemas_to_sample=2,
        target_d=0.7,
        max_cells_per_iteration=1,
        max_items_per_cell=5,
        discovery_model=ModelConfig(
            category="vlm",
            provider="openai",
            model="gpt-4o-mini",
        ),
        vocab_model=ModelConfig(
            category="vlm",
            provider="openai",
            model="gpt-4o-mini",
        ),
        labeling_model=ModelConfig(
            category="vlm",
            provider="openai",
            model="gpt-4o-mini",
        ),
    ),
))

item_type = KnowledgeItemType.ATOMIC
register_experiment(ExperimentConfig(
    task_id=test_task_id,

    name=f"test_{test_task_id}_{item_type.value}",
    description=f"test {test_task_id} {item_type.value} experiment v0.1",
    tags=[f"test", f"{test_task_id}", f"{item_type.value}", "v0.1"],
    constructor=ConstructorConfig(
        use_cache=True,
        use_vision=True,
        cluster_config=ClusterConfig(
            similarity_threshold=0.85,
            model_config=ModelConfig(
                category="embedding",
                provider="openai",
                model="text-embedding-3-small",
            ),
        ),
        model_config=ModelConfig(
            category="vlm",
            provider="openai",
            model="gpt-4o-mini",
        ),
    ),
    discovery=DiscoveryConfig(
        use_cache=True,
        item_type=item_type,
        max_iterations=1,
        k_schemas_to_sample=2,
        target_d=0.7,
        max_cells_per_iteration=1,
        max_items_per_cell=5,
        discovery_model=ModelConfig(
            category="vlm",
            provider="openai",
            model="gpt-4o-mini",
        ),
        vocab_model=ModelConfig(
            category="vlm",
            provider="openai",
            model="gpt-4o-mini",
        ),
        labeling_model=ModelConfig(
            category="vlm",
            provider="openai",
            model="gpt-4o-mini",
        ),
    ),
))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _list_experiments() -> None:
    print(f"\n{'name':<40} {'tags'}")
    print("-" * 70)
    for name, cfg in EXPERIMENTS.items():
        tag_str = ", ".join(cfg.tags) if cfg.tags else "—"
        print(f"{name:<40} [{tag_str}]")
        if cfg.description:
            print(f"  {cfg.description}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run task schema discovery experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--list", action="store_true", help="List all registered experiments.")
    parser.add_argument("--run", metavar="NAME", help="Run an experiment by name.")
    args = parser.parse_args()

    if args.list:
        _list_experiments()
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
