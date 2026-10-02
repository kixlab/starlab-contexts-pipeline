"""
Command-line interface for the contextual-unit pipeline.

    python -m pipeline.cli list
    python -m pipeline.cli run NAME [NAME ...]
    python -m pipeline.cli show TARGET
    python -m pipeline.cli dump TARGET
    python -m pipeline.cli compare TARGET [TARGET ...]
    python -m pipeline.cli budget TARGET [TARGET ...]

TARGET is an experiment name (resolved to its newest saved result with a
matching config) or a path to a ``pipeline_result_*.json`` file. Only ``run``
loads a knowledge base or calls models; the other commands read saved files.
Dataset construction stays in ``python -m dataset``.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from knowledge_items import KnowledgeBase

from .experiments import (
    EXPERIMENTS_DIR,
    ExperimentSpec,
    cached_knowledge_base,
    find_result,
    load_experiments_dir,
    run_experiment,
    summarize,
)
from .models import PipelineResult
from .store import PipelineStore

STATIC_DIR = Path(__file__).parent.parent / "static"


def _context_label(strategy: str, context) -> str:
    """Context components, or "-" for batch runs (which ignore them)."""
    return "-" if strategy == "batch" else "+".join(sorted(context))


# ---------------------------------------------------------------------------
# Target resolution
# ---------------------------------------------------------------------------

def resolve(target: str, experiments_dir: Path) -> Tuple[str, PipelineResult, Optional[ExperimentSpec]]:
    """Return (label, result, spec-or-None) for a name or a result path."""
    path = Path(target)
    if path.suffix == ".json" and path.exists():
        return path.stem, PipelineStore.load_result(path), None
    specs, _ = load_experiments_dir(experiments_dir)
    if target not in specs:
        raise SystemExit(f"Unknown experiment {target!r} and no such result file.")
    spec = specs[target]
    found = find_result(spec)
    if found is None:
        raise SystemExit(f"No saved result for {target!r} with its current config. Run it first.")
    return target, PipelineStore.load_result(found), spec


# ---------------------------------------------------------------------------
# Budget analysis (pure ledger replay)
# ---------------------------------------------------------------------------

def budget_rows(result: PipelineResult) -> List[Dict[str, Any]]:
    """One row per ledger entry: budget spent so far, status, pool tree quality."""
    rows = []
    spent = 0
    for record in result.ledger:
        spent += int(record.counts_against_budget)
        tq = record.tree_quality
        rows.append({
            "attempt": record.attempt_index,
            "budget_spent": spent,
            "status": record.status.value,
            "title": record.title or "",
            "pool_size": len(tq.pool_ids) if tq else 0,
            "tree_nodes": tq.tree_nodes if tq else 0,
            "unique_facets_in_tree": tq.unique_facets_in_tree if tq else 0,
            "nearest_similarity": record.nearest_similarity,
        })
    return rows


def format_budget_table(label: str, result: PipelineResult) -> str:
    header = (f"{'attempt':>7} {'budget':>6} {'status':<18} {'pool':>4} "
              f"{'t_nodes':>7} {'u_facets':>8} {'nn_sim':>6}  title")
    lines = [f"=== Budget analysis: {label} ===", header, "-" * len(header)]
    prev_nodes = None
    for row in budget_rows(result):
        changed = prev_nodes is not None and row["tree_nodes"] != prev_nodes
        prev_nodes = row["tree_nodes"]
        sim = row["nearest_similarity"]
        sim_str = "-" if sim is None else f"{sim:.2f}"
        lines.append(
            f"{row['attempt']:>7} {row['budget_spent']:>6} {row['status']:<18} "
            f"{row['pool_size']:>4} {row['tree_nodes']:>7} "
            f"{row['unique_facets_in_tree']:>8} {sim_str:>6}"
            f"{' *' if changed else '  '}{row['title'][:40]}"
        )
    stop = result.stop_reason.value if result.stop_reason else "none"
    lines.append(f"* = tree_nodes changed | proposals {result.proposals}, "
                 f"attempts {result.attempts}, stop: {stop}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def format_summary(label: str, result: PipelineResult) -> str:
    s = summarize(result)
    cfg = result.config
    lines = [
        f"=== {label} ===",
        f"strategy: {cfg.get('strategy', '?')}  "
        f"context: {_context_label(cfg.get('strategy'), cfg.get('context', []))}  "
        f"budget: {cfg.get('proposal_budget', '?')}  pool_size: {cfg.get('pool_size', '?')}",
        f"stop: {s.stop_reason}  attempts: {s.attempts}  proposals: {s.proposals}  "
        f"units: {s.num_units}",
        f"pool: {s.pool_facets} facets  tree: {s.tree_nodes} nodes, depth {s.tree_depth}",
        "statuses: " + ", ".join(f"{k}={v}" for k, v in sorted(s.status_counts.items())),
        "pool facets:",
    ]
    lines += [f"  - {f.title} ({f.scope}): {f.definition}" for f in result.pool]
    return "\n".join(lines)


def format_comparison(entries: List[Tuple[str, PipelineResult]]) -> str:
    header = (f"{'name':<40} {'strategy':<9} {'context':<22} {'stop':<24} "
              f"{'att':>4} {'prop':>4} {'pool':>4} {'nodes':>5} {'depth':>5}")
    lines = [header, "-" * len(header)]
    for label, result in entries:
        s = summarize(result)
        cfg = result.config
        lines.append(
            f"{label[:40]:<40} {cfg.get('strategy', '?'):<9} "
            f"{_context_label(cfg.get('strategy'), cfg.get('context', [])):<22} "
            f"{s.stop_reason:<24} "
            f"{s.attempts:>4} {s.proposals:>4} {s.pool_facets:>4} "
            f"{s.tree_nodes:>5} {s.tree_depth:>5}"
        )
    return "\n".join(lines)


def format_experiment_list(specs: Dict[str, ExperimentSpec], legacy: List[Path]) -> str:
    lines = [f"{'name':<45} {'strategy':<9} {'context':<22} {'budget':>6} {'pool':>4}  tags"]
    lines.append("-" * len(lines[0]))
    for name, spec in specs.items():
        p = spec.pipeline
        lines.append(
            f"{name:<45} {p.strategy.value:<9} "
            f"{_context_label(p.strategy.value, [c.value for c in p.context]):<22} "
            f"{p.proposal_budget:>6} {p.pool_size:>4}  [{', '.join(spec.tags)}]"
        )
    if legacy:
        lines.append(f"\n{len(legacy)} legacy-format file(s) skipped: "
                     + ", ".join(p.name for p in legacy))
    return "\n".join(lines)


def write_facets_csv(result: PipelineResult, path: Path,
                     knowledge_base: Optional[KnowledgeBase] = None,
                     max_examples: int = 3) -> Path:
    """One row per pool facet; label examples included when a KB is given."""
    examples: Dict[str, Dict[str, List[str]]] = {}
    if knowledge_base is not None:
        items = {item.id: item for item in knowledge_base}
        for (item_id, facet_id), label in result.annotations.items():
            if item_id in items:
                examples.setdefault(facet_id, {}).setdefault(label, []).append(
                    items[item_id].text
                )

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Facet ID", "Details"])
        for facet in result.pool:
            lines = [f"**{facet.title}** ({facet.scope})", "",
                     f"**Definition:** {facet.definition}", ""]
            if facet.guidelines:
                lines.append("**Guidelines:**")
                lines += [f"- {g}" for g in facet.guidelines]
                lines.append("")
            if facet.vocabulary:
                lines.append("**Values:**")
                for v in facet.vocabulary:
                    lines.append(f"- **{v.label}**: {v.definition}")
                    for ex in examples.get(facet.id, {}).get(v.label, [])[:max_examples]:
                        lines.append(f"    - e.g. \"{ex}\"")
            writer.writerow([facet.id, "\n".join(lines)])
    return path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None, experiments_dir: Path = EXPERIMENTS_DIR) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m pipeline.cli", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="List experiment configurations.")
    p_run = sub.add_parser("run", help="Run experiments (calls models).")
    p_run.add_argument("names", nargs="+")
    p_show = sub.add_parser("show", help="Show a saved result.")
    p_show.add_argument("target")
    p_dump = sub.add_parser("dump", help="Dump a result's pool facets to static/*.csv.")
    p_dump.add_argument("target")
    p_cmp = sub.add_parser("compare", help="Compare saved results.")
    p_cmp.add_argument("targets", nargs="+")
    p_bud = sub.add_parser("budget", help="Replay a result's candidate ledger.")
    p_bud.add_argument("targets", nargs="+")
    args = parser.parse_args(argv)

    if args.command == "list":
        print(format_experiment_list(*load_experiments_dir(experiments_dir)))
    elif args.command == "run":
        logging.basicConfig(level=logging.INFO,
                            format="%(levelname)s %(name)s: %(message)s", force=True)
        specs, _ = load_experiments_dir(experiments_dir)
        missing = [n for n in args.names if n not in specs]
        if missing:
            raise SystemExit(f"Unknown experiment(s): {missing}")
        for name in args.names:
            print(format_summary(name, run_experiment(specs[name])))
    elif args.command == "show":
        label, result, _ = resolve(args.target, experiments_dir)
        print(format_summary(label, result))
    elif args.command == "dump":
        label, result, spec = resolve(args.target, experiments_dir)
        kb = cached_knowledge_base(spec) if spec is not None else None
        path = write_facets_csv(result, STATIC_DIR / f"facets_{label}.csv", kb)
        print(f"Wrote {len(result.pool)} facets to {path}")
    elif args.command == "compare":
        entries = [resolve(t, experiments_dir)[:2] for t in args.targets]
        print(format_comparison(entries))
    elif args.command == "budget":
        for target in args.targets:
            label, result, _ = resolve(target, experiments_dir)
            print(format_budget_table(label, result))
            print()


if __name__ == "__main__":
    main(sys.argv[1:])
