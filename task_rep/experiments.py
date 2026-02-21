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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .algorithm import DiscoveryConfig, DiscoveryResult, ExtractionConfig, IterativeSchemaDiscovery, ModelConfig
from .facets.io import load_annotations, save_annotations, save_facets
from .knowledge_items import (
    ClusterConfig,
    KnowledgeItemType,
    extract_and_cluster_knowledge_items,
    load_dataset,
)
from .optimization.metrics import count_unique_units


@dataclass
class DatasetConfig:
    """Which dataset to use and how many videos to include."""

    task_id: str
    """Key in static/tasks.json (e.g. 'sushi_roll_3')."""

    dataset_path: str
    """Path to the built dataset JSON (e.g. 'task_rep/static/datasets/sushi_roll_3.json')."""

    max_videos: Optional[int] = None
    """Limit to first N videos. None = all videos in the file."""


@dataclass
class ExperimentConfig:
    """Full specification of one experiment run."""

    name: str
    """Short identifier used as a key in EXPERIMENTS and for file naming."""

    dataset: DatasetConfig
    extraction: ExtractionConfig
    discovery: DiscoveryConfig

    description: str = ""
    """Free-text notes about what this experiment is testing."""

    tags: List[str] = field(default_factory=list)
    """Labels for filtering/grouping (e.g. ['baseline', 'sushi', 'atomic'])."""


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

    best_discriminativeness: float
    """Lowest D value on the frontier (0 = perfect separation)."""

    best_compactness: int
    """Vocabulary size of the schema with the best D."""

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
    discovery: DiscoveryResult
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

    Pipeline:
      1. Load dataset videos (optionally capped at max_videos).
      2. Extract + cluster knowledge items → KnowledgeBase.
      3. Run IterativeSchemaDiscovery.

    Args:
        config: Full experiment specification.
        verbose: Enable INFO-level logging and progress prints.

    Returns:
        ExperimentResult with the discovery output and run metadata.
    """
    import logging

    if verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(levelname)s %(name)s: %(message)s",
            force=True,
        )

    t0 = time.monotonic()

    # Step 1 — load videos
    videos = load_dataset(config.dataset.dataset_path)
    if config.dataset.max_videos is not None:
        videos = videos[: config.dataset.max_videos]

    if verbose:
        print(f"\n[{config.name}] {len(videos)} video(s) from '{config.dataset.task_id}'")

    # Step 2 — extract + cluster knowledge items
    kb = extract_and_cluster_knowledge_items(
        task_id=config.dataset.task_id,
        granularity=config.extraction.granularity,
        videos=videos,
        base_dir=config.extraction.base_dir,
        provider=config.extraction.provider,
        model=config.extraction.model,
        cluster_config=config.extraction.cluster_config,
        use_cache=config.extraction.use_cache,
        max_concurrency=config.extraction.max_concurrency,
    )

    num_units = count_unique_units(kb)
    if verbose:
        print(f"[{config.name}] KnowledgeBase: {len(kb)} items, {num_units} units")

    # Step 3 — run discovery
    discovery = IterativeSchemaDiscovery(config.discovery).run(kb)

    elapsed = time.monotonic() - t0

    summary = ExperimentSummary(
        num_units=num_units,
        num_facets=len(discovery.facets),
        frontier_size=len(discovery.frontier),
        best_discriminativeness=(
            discovery.frontier[0].discriminativeness if discovery.frontier else float("inf")
        ),
        best_compactness=(
            discovery.frontier[0].compactness if discovery.frontier else 0
        ),
        iterations=discovery.iterations,
    )

    return ExperimentResult(
        run_id=str(uuid.uuid4())[:8],
        timestamp=datetime.now(timezone.utc).isoformat(),
        config=config,
        discovery=discovery,
        summary=summary,
        elapsed_seconds=elapsed,
    )


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def print_summary(result: ExperimentResult) -> None:
    """Print a human-readable one-page summary of an experiment result."""
    s = result.summary
    c = result.config
    d = c.discovery
    sep = "=" * 60
    print(sep)
    print(f"  {c.name}  [{result.run_id}]")
    if c.description:
        print(f"  {c.description}")
    print(sep)
    print(f"  dataset      : {c.dataset.task_id}")
    print(f"  max_videos   : {c.dataset.max_videos or 'all'}")
    print(f"  granularity  : {c.extraction.granularity.value}")
    print(f"  cluster sim  : {c.extraction.cluster_config.similarity_threshold}")
    print(f"  provider     : {d.provider or 'default'}  /  model={d.model or 'default'}")
    print(f"  item_type    : {d.item_type.value if d.item_type else 'all'}")
    print(f"  tags         : {c.tags or '—'}")
    print(f"  ---")
    print(f"  units        : {s.num_units}")
    print(f"  facets found : {s.num_facets}")
    print(f"  frontier pts : {s.frontier_size}")
    print(f"  best D       : {s.best_discriminativeness:.4f}  (C={s.best_compactness})")
    print(f"  iterations   : {s.iterations}")
    print(f"  elapsed      : {result.elapsed_seconds:.1f}s")
    print(f"  timestamp    : {result.timestamp}")
    print()


def compare_results(results: List[ExperimentResult]) -> None:
    """Print a compact side-by-side comparison table."""
    cols = f"{'name':<38} {'units':>6} {'facets':>7} {'best_D':>7} {'best_C':>7} {'iters':>6} {'secs':>6}"
    print(cols)
    print("-" * len(cols))
    for r in results:
        s = r.summary
        print(
            f"{r.config.name:<38} {s.num_units:>6} {s.num_facets:>7}"
            f" {s.best_discriminativeness:>7.4f} {s.best_compactness:>7}"
            f" {s.iterations:>6} {r.elapsed_seconds:>6.1f}"
        )


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert a (possibly nested) config/summary to JSON-safe types."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, KnowledgeItemType):
        return obj.value
    if isinstance(obj, list):
        return [_to_jsonable(i) for i in obj]
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    return obj


def save_result(result: ExperimentResult, save_dir: str | Path) -> Path:
    """
    Persist an ExperimentResult to disk.

    Creates:
        {save_dir}/{name}_{run_id}/
            meta.json         — config snapshot, summary metrics, timing
            facets.json       — final facet pool (Facet objects)
            annotations.json  — full annotation table
            frontier.json     — Pareto frontier points

    Returns:
        Path to the created run directory.
    """
    run_dir = Path(save_dir) / f"{result.config.name}_{result.run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # meta: config + summary + timing (no heavy data)
    meta: Dict[str, Any] = {
        "run_id": result.run_id,
        "timestamp": result.timestamp,
        "elapsed_seconds": result.elapsed_seconds,
        "config": _to_jsonable(result.config),
        "summary": _to_jsonable(result.summary),
    }
    (run_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # facets, annotations
    save_facets(result.discovery.facets, run_dir / "facets.json")
    save_annotations(result.discovery.annotations, run_dir / "annotations.json")

    # frontier
    frontier_data = [
        {
            "discriminativeness": fp.discriminativeness,
            "compactness": fp.compactness,
            "facet_ids": fp.facet_ids,
        }
        for fp in result.discovery.frontier
    ]
    (run_dir / "frontier.json").write_text(
        json.dumps(frontier_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Saved to {run_dir}/")
    return run_dir


def load_result_meta(run_dir: str | Path) -> Dict[str, Any]:
    """
    Load only the meta.json from a saved result directory.

    Useful for building comparison tables without loading full facets/annotations.
    """
    return json.loads(Path(run_dir).joinpath("meta.json").read_text(encoding="utf-8"))


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

register_experiment(ExperimentConfig(
    name="sushi_atomic_baseline",
    description="Sushi roll (3 videos) · ATOMIC items · default model (gpt-4o-mini)",
    tags=["baseline", "sushi", "atomic"],
    dataset=DatasetConfig(
        task_id="sushi_roll_3",
        dataset_path="task_rep/static/datasets/sushi_roll_3.json",
    ),
    extraction=ExtractionConfig(
        granularity=KnowledgeItemType.ATOMIC,
        cluster_config=ClusterConfig(similarity_threshold=0.85),
    ),
    discovery=DiscoveryConfig(
        task="making a sushi roll",
        item_type=KnowledgeItemType.ATOMIC,
        base_dir="task_rep/static/cache",
        task_id="sushi_roll_3",
    ),
))

register_experiment(ExperimentConfig(
    name="sushi_step_baseline",
    description="Sushi roll (3 videos) · STEP items · default model",
    tags=["baseline", "sushi", "step"],
    dataset=DatasetConfig(
        task_id="sushi_roll_3",
        dataset_path="task_rep/static/datasets/sushi_roll_3.json",
    ),
    extraction=ExtractionConfig(
        granularity=KnowledgeItemType.STEP,
        cluster_config=ClusterConfig(similarity_threshold=0.85),
    ),
    discovery=DiscoveryConfig(
        task="making a sushi roll",
        item_type=KnowledgeItemType.STEP,
        base_dir="task_rep/static/cache",
        task_id="sushi_roll_3",
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
    parser.add_argument(
        "--save-dir",
        default=None,
        metavar="DIR",
        help="If provided, save the result to this directory.",
    )
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
        print_summary(result)
        if args.save_dir:
            save_result(result, args.save_dir)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
