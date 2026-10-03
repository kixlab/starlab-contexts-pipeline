"""
Reporting over saved run results, shared by every runner's CLI.

Summaries, ledger-based budget analysis, token usage, refinement-threshold
sweeps, and facet CSV export. Everything except ``tau_sweep`` (which reads a
knowledge base) works from a ``RunResult`` alone.
"""

from __future__ import annotations

import csv
import dataclasses
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from knowledge_items import KnowledgeBase

from optimization import compute_tree_depth

from .corpus import select_corpus
from .evaluation import PoolEvaluator
from .models import RunResult
from .settings import RunSettings


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

@dataclass
class RunSummary:
    """Key metrics for quick comparison across runs."""

    num_units: int
    pool_facets: int
    iterations: int
    attempts: int
    proposals: int
    tree_nodes: int
    tree_depth: int
    stop_reason: str
    status_counts: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def summarize(result: RunResult) -> RunSummary:
    counts = Counter(r.status.value for r in result.ledger)
    return RunSummary(
        num_units=result.num_units,
        pool_facets=len(result.pool),
        iterations=result.iterations,
        attempts=result.attempts,
        proposals=result.proposals,
        tree_nodes=len(result.tree.nodes) if result.tree else 0,
        tree_depth=compute_tree_depth(result.tree) if result.tree else 0,
        stop_reason=result.stop_reason.value if result.stop_reason else "none",
        status_counts=dict(counts),
    )


def format_result_body(result: RunResult) -> List[str]:
    """Summary lines after a runner-specific header line."""
    s = summarize(result)
    lines = [
        f"stop: {s.stop_reason}  attempts: {s.attempts}  proposals: {s.proposals}  "
        f"units: {s.num_units}",
        f"pool: {s.pool_facets} facets  tree: {s.tree_nodes} nodes, depth {s.tree_depth}",
        "statuses: " + ", ".join(f"{k}={v}" for k, v in sorted(s.status_counts.items())),
        *format_usage(result),
        "pool facets:",
    ]
    lines += [f"  - {f.title} ({f.scope}): {f.definition}" for f in result.pool]
    return lines


# ---------------------------------------------------------------------------
# Token usage
# ---------------------------------------------------------------------------

def total_usage(result: RunResult) -> Dict[str, int]:
    keys = ("input_tokens", "output_tokens", "thinking_tokens", "requests")
    return {k: sum(stage.get(k, 0) for stage in result.usage.values()) for k in keys}


def format_usage(result: RunResult) -> List[str]:
    if not result.usage:
        return ["usage: not recorded"]
    lines = [f"usage ({'stage':<10} {'requests':>8} {'input':>12} {'output':>10} {'thinking':>10}):"]
    rows = sorted(result.usage.items()) + [("total", total_usage(result))]
    for stage, u in rows:
        lines.append(
            f"  {stage:<16} {u.get('requests', 0):>8} {u.get('input_tokens', 0):>12,} "
            f"{u.get('output_tokens', 0):>10,} {u.get('thinking_tokens', 0):>10,}"
        )
    return lines


# ---------------------------------------------------------------------------
# Budget analysis (pure ledger replay)
# ---------------------------------------------------------------------------

def budget_rows(result: RunResult) -> List[Dict[str, Any]]:
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


def format_budget_table(label: str, result: RunResult) -> str:
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
# Refinement-threshold sweep
# ---------------------------------------------------------------------------

DEFAULT_TAUS = [0.2, 0.3, 0.4, 0.5, 0.6]


def tau_sweep(result: RunResult, knowledge_base: KnowledgeBase,
              config: RunSettings, taus: List[float]) -> List[Dict[str, Any]]:
    """Rebuild the result's tree at each refinement threshold (no model calls).

    Applies the run's corpus selection, so the tree covers the same videos.
    """
    focused = select_corpus(knowledge_base, config)
    if config.item_scope.atomic_subtype.subtypes:
        focused = focused.filter_subtypes(config.item_scope.atomic_subtype.subtypes)
    rows = []
    for tau in taus:
        evaluator = PoolEvaluator(
            config=dataclasses.replace(config, refinement_threshold=tau), focused_kb=focused,
        )
        evaluation = evaluator.evaluate_pool(result.pool, result.annotations)
        tree = evaluation.tree
        rows.append({
            "tau": tau,
            "tree_nodes": len(tree.nodes),
            "tree_depth": compute_tree_depth(tree),
            "unique_facets_in_tree": len({fid for n in tree.nodes for fid in n.facet_ids}),
            "rejected_schemas": len(evaluation.rejected_schemas),
        })
    return rows


def format_tau_table(label: str, result: RunResult, rows: List[Dict[str, Any]]) -> str:
    lines = [f"=== Tree by refinement threshold: {label} ({len(result.pool)} pool facets) ===",
             f"{'tau':>5} {'t_nodes':>7} {'depth':>5} {'u_facets':>8} {'rejected':>8}"]
    for r in rows:
        lines.append(f"{r['tau']:>5.2f} {r['tree_nodes']:>7} {r['tree_depth']:>5} "
                     f"{r['unique_facets_in_tree']:>8} {r['rejected_schemas']:>8}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Facet export
# ---------------------------------------------------------------------------

def write_facets_csv(result: RunResult, path: Path,
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
