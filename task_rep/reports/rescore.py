"""
Re-score finished runs under changed evaluation settings, without model calls.

    python -m reports.rescore pipeline sushi_roll_step_pilot15_a0 \\
        --set refinement_threshold=0.55 --set max_schema_facets=2

Loads the newest result under the experiment file's current settings,
re-evaluates its tree and per-decision snapshots under the overrides
(``analysis.rescore``), and saves it as a new result under the new settings.
Then update the experiment file to the new settings so ``find_result`` picks
it up (it is a new run folder; the original run is untouched). Refuses runs whose pool filled up, or whose proposer saw the tree.
"""

from __future__ import annotations

import argparse
import datetime
from typing import List, Optional

from analysis.rescore import rescore_evaluation

from .runs import load_run, parse_overrides, save_run

EVALUATION_FIELDS = {"refinement_threshold", "max_schema_facets", "metric_mode",
                     "eta_squared_weighting"}


def _sees_tree(kind: str, settings) -> bool:
    if kind == "pipeline":
        return "schemas" in {c.value for c in getattr(settings, "context", [])}
    return getattr(settings, "feedback", None) is not None and settings.feedback.value == "schemas"


def rescore(kind: str, experiment: str, overrides: dict, force: bool = False):
    unknown = set(overrides) - EVALUATION_FIELDS
    if unknown:
        raise SystemExit(f"only evaluation settings can be re-scored, not {sorted(unknown)}")
    run = load_run(kind, experiment)
    if run.settings.pool_size < run.settings.proposal_budget and not force:
        raise SystemExit("the pool could fill up, so selection depended on the tree; re-run instead")
    if _sees_tree(kind, run.settings) and not force:
        raise SystemExit("this condition's proposer saw the tree; re-run instead")
    state = run.result.to_state()
    new = run.settings.__class__.from_dict({**run.settings.to_dict(), **overrides})
    note = {"date": datetime.date.today().isoformat(),
            "rules": ", ".join(f"{k} {getattr(run.settings, k)} -> {v}" for k, v in overrides.items()),
            "original_result": run.result.fingerprint}
    rescore_evaluation(state, run.focused, new, note)
    path = save_run(kind, experiment, new, run.corpus, run.path.parent, state, run.focused)
    tree = state.tree
    print(f"{experiment}: tree {len(run.result.tree.nodes) if run.result.tree else 0} -> "
          f"{len(tree.nodes) if tree else 0} nodes; saved {path.name}")
    return path


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("kind", choices=["pipeline", "agentic"])
    parser.add_argument("experiments", nargs="+", help="experiment names (file stems)")
    parser.add_argument("--set", dest="overrides", action="append", required=True,
                        metavar="KEY=VALUE", help="evaluation setting to change")
    parser.add_argument("--force", action="store_true",
                        help="re-score even if the run could have depended on the tree")
    args = parser.parse_args(argv)
    overrides = parse_overrides(args.overrides)
    for experiment in args.experiments:
        rescore(args.kind, experiment, overrides, args.force)


if __name__ == "__main__":
    main()
