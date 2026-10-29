"""
Command-line interface for the contextual-unit pipeline.

    python -m pipeline.cli list
    python -m pipeline.cli run NAME [NAME ...] [--fresh]
    python -m pipeline.cli show TARGET
    python -m pipeline.cli dump TARGET
    python -m pipeline.cli compare TARGET [TARGET ...]
    python -m pipeline.cli budget TARGET [TARGET ...]
    python -m pipeline.cli tree NAME [--tau T ...]

``run`` resumes an experiment's interrupted run, or starts a new one
(``--fresh`` always starts a new one). TARGET is an experiment name (its newest
finished run with a matching config) or a run folder
(``static/cache_2/<task>/runs/<time>__<experiment>``). Only ``run`` calls models; ``tree`` reads the experiment's cached knowledge base; the other
commands read saved result files only.
Dataset construction stays in ``python -m dataset``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from contextual.models import RunResult
from contextual.reporting import (
    DEFAULT_TAUS,
    format_budget_table,
    format_result_body,
    format_tau_table,
    summarize,
    tau_sweep,
    total_usage,
    write_facets_csv,
)
from contextual.cache import load_result

from .experiments import (
    EXPERIMENTS_DIR,
    ExperimentSpec,
    cached_knowledge_base,
    find_result,
    load_experiments_dir,
    run_experiment,
)

STATIC_DIR = Path(__file__).parent.parent / "static"


def _context_label(strategy: str, context) -> str:
    """Context components, or "-" for batch runs (which ignore them)."""
    return "-" if strategy == "batch" else "+".join(sorted(context))


# ---------------------------------------------------------------------------
# Target resolution
# ---------------------------------------------------------------------------

def resolve(target: str, experiments_dir: Path) -> Tuple[str, RunResult, Optional[ExperimentSpec]]:
    """Return (label, result, spec-or-None) for a name or a result path."""
    path = Path(target)
    if path.exists():                                   # a run folder or its result.json
        folder = path if path.is_dir() else path.parent
        print(f"[run {folder}]", file=sys.stderr)
        return folder.name, load_result(path), None
    specs, _ = load_experiments_dir(experiments_dir)
    if target not in specs:
        raise SystemExit(f"Unknown experiment {target!r} and no such run folder.")
    spec = specs[target]
    found = find_result(spec)
    if found is None:
        raise SystemExit(f"No saved result for {target!r} with its current config. Run it first.")
    print(f"[run {found}]", file=sys.stderr)
    return target, load_result(found), spec


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def format_summary(label: str, result: RunResult) -> str:
    cfg = result.config
    lines = [
        f"=== {label} ===",
        f"strategy: {cfg.get('strategy', '?')}  "
        f"context: {_context_label(cfg.get('strategy'), cfg.get('context', []))}  "
        f"budget: {cfg.get('proposal_budget', '?')}  pool_size: {cfg.get('pool_size', '?')}",
        *format_result_body(result),
    ]
    return "\n".join(lines)


def format_comparison(entries: List[Tuple[str, RunResult]]) -> str:
    header = (f"{'name':<40} {'strategy':<9} {'context':<22} {'stop':<24} "
              f"{'att':>4} {'prop':>4} {'pool':>4} {'nodes':>5} {'depth':>5} "
              f"{'in_tok':>11} {'think_tok':>10}")
    lines = [header, "-" * len(header)]
    for label, result in entries:
        s = summarize(result)
        cfg = result.config
        lines.append(
            f"{label[:40]:<40} {cfg.get('strategy', '?'):<9} "
            f"{_context_label(cfg.get('strategy'), cfg.get('context', [])):<22} "
            f"{s.stop_reason:<24} "
            f"{s.attempts:>4} {s.proposals:>4} {s.pool_facets:>4} "
            f"{s.tree_nodes:>5} {s.tree_depth:>5} "
            f"{total_usage(result)['input_tokens']:>11,} "
            f"{total_usage(result)['thinking_tokens']:>10,}"
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
    p_run.add_argument("--fresh", action="store_true",
                       help="start a new run even if the last one was interrupted")
    p_show = sub.add_parser("show", help="Show a saved result.")
    p_show.add_argument("target")
    p_dump = sub.add_parser("dump", help="Dump a result's pool facets to static/*.csv.")
    p_dump.add_argument("target")
    p_cmp = sub.add_parser("compare", help="Compare saved results.")
    p_cmp.add_argument("targets", nargs="+")
    p_bud = sub.add_parser("budget", help="Replay a result's candidate ledger.")
    p_bud.add_argument("targets", nargs="+")
    p_tree = sub.add_parser("tree", help="Rebuild a result's tree at several taus.")
    p_tree.add_argument("name", help="experiment name (needs its cached KB)")
    p_tree.add_argument("--tau", type=float, nargs="+", default=DEFAULT_TAUS)
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
            print(format_summary(name, run_experiment(specs[name], fresh=args.fresh)))
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
    elif args.command == "tree":
        label, result, spec = resolve(args.name, experiments_dir)
        if spec is None:
            raise SystemExit("tree needs an experiment name (for its knowledge base).")
        kb = cached_knowledge_base(spec)
        if kb is None:
            raise SystemExit(f"Knowledge base for {label!r} is not cached.")
        print(format_tau_table(label, result, tau_sweep(result, kb, spec.pipeline, args.tau)))


if __name__ == "__main__":
    main(sys.argv[1:])
