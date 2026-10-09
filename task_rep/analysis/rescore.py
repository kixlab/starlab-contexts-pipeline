"""
Re-score a finished run under new evaluation settings (no model calls).

Only valid when the change cannot alter what happened during the run: the
pool must never have filled (``pool_size >= proposal_budget``, so selection
did not depend on the tree) and the proposer must not have seen the tree.
Then the pool and ledger stay as they are; the final tree and every
per-decision tree snapshot are re-evaluated under ``settings``.
"""

from __future__ import annotations

from typing import Dict

from contextual.bookkeeping import snapshot
from contextual.evaluation import PoolEvaluator
from contextual.models import RunState
from contextual.settings import RunSettings
from knowledge_items import KnowledgeBase


def rescore_evaluation(state: RunState, focused_kb: KnowledgeBase, settings: RunSettings,
                       note: Dict[str, str]) -> RunState:
    """Re-evaluate ``state``'s tree and ledger snapshots in place.

    ``note`` is stored under ``runner_state["rescored"]`` (e.g. date, rules,
    original result fingerprint).
    """
    evaluator = PoolEvaluator(config=settings, focused_kb=focused_kb)
    by_id = {f.id: f for f in state.pool}
    for record in state.ledger:
        snap = record.tree_quality
        if snap is None:
            continue
        missing = [fid for fid in snap.pool_ids if fid not in by_id]
        if missing:
            raise ValueError(f"snapshot pool {missing} is not in the final pool; the pool "
                             "changed during the run, so it cannot be re-scored")
        trial = RunState(pool=[by_id[fid] for fid in snap.pool_ids],
                         annotations=state.annotations.filter_facets(snap.pool_ids))
        if trial.pool:
            evaluator.apply_evaluation(trial, evaluator.evaluate_pool(trial.pool, trial.annotations))
        record.tree_quality = snapshot(trial)
    if state.pool:
        evaluator.apply_evaluation(state, evaluator.evaluate_pool(state.pool, state.annotations))
    note = dict(note)
    previous = state.runner_state.get("rescored") or state.runner_state.get("rescored_dag")
    if previous:
        note["previous"] = previous
    state.runner_state = {**state.runner_state, "rescored": note}
    return state
