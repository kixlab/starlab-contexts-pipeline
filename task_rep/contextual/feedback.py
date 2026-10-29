"""
Feedback inputs shared by every runner: the history of tried factors that are
not in the pool, and what the latest submission did to the design space.

Both are plain data; rendering lives in ``facets.prompts.formatters``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .models import CandidateRecord, CandidateStatus, RunState

# Ledger records that never describe a tried factor.
NO_HISTORY_STATUSES = {
    CandidateStatus.REFUSED,
    CandidateStatus.INVALID,
    CandidateStatus.TRUNCATED,
}
# Statuses whose factor entered the pool, and so the design space.
IN_POOL_STATUSES = {CandidateStatus.ACCEPTED, CandidateStatus.REPLACED}


@dataclass
class HistoryEntry:
    """A previously tried candidate that is not in the pool (HISTORY).

    ``outcome`` is the candidate's ledger status value, or ``"replaced_out"`` for
    a facet that was accepted and later evicted from the pool.
    """

    title: str
    scope: str
    definition: str
    outcome: str
    duplicate_of: Optional[str] = None  # title of the matched candidate


@dataclass
class LastSubmission:
    """The latest ledger record, for the design-space feedback.

    ``previous_schema_ids`` are the accepted schema ids before it was
    evaluated; ``entered`` says whether its factor is now in the design space.
    """

    title: str
    facet_id: str
    outcome: str
    entered: bool
    previous_schema_ids: List[str] = field(default_factory=list)


def _titles(state: RunState) -> Dict[str, str]:
    return {r.facet_id: r.title for r in state.ledger if r.facet_id and r.title}


def history_entries(state: RunState) -> List[HistoryEntry]:
    """Every tried factor that is not in the pool, in ledger order."""
    pool_ids = set(state.pool_ids())
    titles = _titles(state)
    history = []
    for r in state.ledger:
        if r.status in NO_HISTORY_STATUSES or r.facet_id in pool_ids:
            continue
        outcome = "replaced_out" if r.status in IN_POOL_STATUSES else r.status.value
        history.append(HistoryEntry(
            title=r.title or "", scope=r.scope or "", definition=r.definition or "",
            outcome=outcome,
            duplicate_of=titles.get(r.duplicate_of) if r.duplicate_of else None,
        ))
    return history


def describe_outcome(entry: CandidateRecord, state: RunState) -> str:
    """One plain sentence for a ledger record's status."""
    titles = _titles(state)
    s = entry.status
    if s == CandidateStatus.ACCEPTED:
        return "accepted into the pool"
    if s == CandidateStatus.REPLACED:
        return f"accepted; it replaced \"{titles.get(entry.replaced_facet_id, '?')}\" in the pool"
    if s == CandidateStatus.DUPLICATE:
        return (f"rejected as a duplicate of the earlier factor "
                f"\"{titles.get(entry.duplicate_of, '?')}\"")
    if s == CandidateStatus.IRR_PRUNED:
        return "rejected; independent labelers disagreed too often (unreliable labeling)"
    if s == CandidateStatus.VOCABULARY_FAILED:
        return "rejected; no usable vocabulary could be built from the tutorials"
    if s == CandidateStatus.LABELING_FAILED:
        return "rejected; the tutorial segments could not be labeled"
    if s == CandidateStatus.SCHEMA_REJECTED:
        return "rejected; it did not improve the design space enough to replace a pooled factor"
    if s == CandidateStatus.REFUSED:
        return "no factor was proposed"
    if s == CandidateStatus.INVALID:
        return "the proposal could not be read"
    return s.value


def last_submission(
    state: RunState, entry: Optional[CandidateRecord] = None,
) -> Optional[LastSubmission]:
    """The latest submission (``entry``, default the last ledger record)."""
    entry = entry if entry is not None else (state.ledger[-1] if state.ledger else None)
    if entry is None or entry.status == CandidateStatus.TRUNCATED:
        return None
    return LastSubmission(
        title=entry.title or "",
        facet_id=entry.facet_id or "",
        outcome=describe_outcome(entry, state),
        entered=entry.status in IN_POOL_STATUSES and entry.facet_id in set(state.pool_ids()),
        previous_schema_ids=list(state.schema_ids_before_last),
    )
