"""
Ledger and counter bookkeeping shared by every runner (the normative counter
table in DESIGN.md).

Each candidate outcome is recorded once with ``record``; ``apply_outcome`` runs
pool selection for a processed candidate first. ``stop_reason`` evaluates the
shared bounds in precedence order.
"""

from __future__ import annotations

from typing import Optional

from facets import Facet

from .evaluation import PoolEvaluator
from .lifecycle import CandidateFailure, CandidateOutcome, ProcessedCandidate
from .models import (
    CandidateRecord, CandidateStatus, RunState, StopReason, TreeQualitySnapshot,
)
from .settings import RunSettings

# Statuses whose candidate entered the lifecycle (passed dedup). Their
# embeddings join the dedup set; duplicates never do, so near-copies cannot chain.
LIFECYCLE_STATUSES = {
    CandidateStatus.IRR_PRUNED,
    CandidateStatus.VOCABULARY_FAILED,
    CandidateStatus.LABELING_FAILED,
    CandidateStatus.SCHEMA_REJECTED,
    CandidateStatus.ACCEPTED,
    CandidateStatus.REPLACED,
}
# Statuses that count against the proposal budget: every parseable proposal,
# however early it is dropped. Refusals (nothing proposed), invalid
# (unparseable) proposals, and a0 truncation surplus do not count.
BUDGET_STATUSES = LIFECYCLE_STATUSES | {CandidateStatus.DUPLICATE}
# Statuses that increment the no-improvement streak (processed but no gain).
NO_IMPROVE_STATUSES = {
    CandidateStatus.DUPLICATE,
    CandidateStatus.IRR_PRUNED,
    CandidateStatus.VOCABULARY_FAILED,
    CandidateStatus.LABELING_FAILED,
    CandidateStatus.SCHEMA_REJECTED,
}


def snapshot(state: RunState) -> TreeQualitySnapshot:
    tree = state.tree
    nodes = len(tree.nodes) if tree else 0
    unique = len({fid for n in (tree.nodes if tree else []) for fid in n.facet_ids})
    return TreeQualitySnapshot(nodes, unique, state.pool_ids())


def record(
    state: RunState, status: CandidateStatus, *,
    increment_attempts: bool = True,
    facet: Optional[Facet] = None,
    nearest_id: Optional[str] = None,
    nearest_similarity: Optional[float] = None,
    duplicate_of: Optional[str] = None,
    replaced_facet_id: Optional[str] = None,
    tree_quality: Optional[TreeQualitySnapshot] = None,
    note: str = "",
) -> CandidateRecord:
    """Update counters for one outcome and append its ledger record."""
    if status == CandidateStatus.TRUNCATED:
        # a0 surplus: recorded for completeness, touches no counter.
        increment_attempts = False
    else:
        state.refusal_streak = (state.refusal_streak + 1
                                if status == CandidateStatus.REFUSED else 0)
        if status in (CandidateStatus.ACCEPTED, CandidateStatus.REPLACED):
            state.no_improve_streak = 0
        elif status in NO_IMPROVE_STATUSES:
            state.no_improve_streak += 1
        # REFUSED / INVALID leave no_improve_streak unchanged
    if increment_attempts:
        state.attempts += 1
    counts = status in BUDGET_STATUSES
    if counts:
        state.proposals += 1

    entry = CandidateRecord(
        attempt_index=state.attempts,
        status=status,
        facet_id=facet.id if facet else None,
        title=facet.title if facet else None,
        scope=facet.scope if facet else None,
        definition=facet.definition if facet else None,
        counts_against_budget=counts,
        nearest_id=nearest_id,
        nearest_similarity=nearest_similarity,
        duplicate_of=duplicate_of,
        replaced_facet_id=replaced_facet_id,
        tree_quality=tree_quality if tree_quality is not None else snapshot(state),
        note=note,
    )
    state.ledger.append(entry)
    return entry


def apply_outcome(
    state: RunState, evaluator: PoolEvaluator, outcome: CandidateOutcome, *,
    increment_attempts: bool,
) -> CandidateRecord:
    """Run pool selection for a processed candidate; record any outcome."""
    if isinstance(outcome, ProcessedCandidate):
        state.embeddings[outcome.facet.id] = outcome.embedding
        decision = evaluator.consider(state, outcome)
        return record(
            state, decision.status, increment_attempts=increment_attempts,
            facet=outcome.facet, nearest_id=outcome.nearest_id,
            nearest_similarity=outcome.nearest_similarity,
            replaced_facet_id=decision.replaced_facet_id,
            tree_quality=decision.snapshot, note=outcome.note,
        )

    failure: CandidateFailure = outcome
    if failure.status in LIFECYCLE_STATUSES and failure.embedding is not None:
        state.embeddings[failure.facet.id] = failure.embedding
    return record(
        state, failure.status, increment_attempts=increment_attempts,
        facet=failure.facet, nearest_id=failure.nearest_id,
        nearest_similarity=failure.nearest_similarity,
        duplicate_of=failure.duplicate_of, note=failure.note,
    )


def stop_reason(state: RunState, settings: RunSettings) -> Optional[StopReason]:
    """First tripped shared bound, in precedence order."""
    if state.proposals >= settings.proposal_budget:
        return StopReason.PROPOSAL_BUDGET
    if state.attempts >= settings.attempt_limit:
        return StopReason.ATTEMPT_LIMIT
    if state.refusal_streak >= settings.refusal_patience:
        return StopReason.REFUSAL_PATIENCE
    if state.no_improve_streak >= settings.no_improvement_patience:
        return StopReason.NO_IMPROVEMENT_PATIENCE
    return None
