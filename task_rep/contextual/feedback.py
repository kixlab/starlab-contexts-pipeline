"""
Feedback inputs shared by every runner: stable factor ids and the history of
tried factors that are not in the pool.

Both are plain data; rendering lives in ``facets.prompts.formatters``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .models import CandidateRecord, CandidateStatus, RunState

# Ledger records that never describe a tried factor.
NO_HISTORY_STATUSES = {
    CandidateStatus.REFUSED,
    CandidateStatus.INVALID,
    CandidateStatus.TRUNCATED,
}
# Statuses whose factor entered the pool.
IN_POOL_STATUSES = {CandidateStatus.ACCEPTED, CandidateStatus.REPLACED}


@dataclass
class HistoryEntry:
    """A previously tried candidate that is not in the pool (HISTORY).

    ``outcome`` says in plain words why it is not in the pool.
    """

    title: str
    scope: str
    definition: str
    outcome: str


def factor_ids(state: RunState) -> Dict[str, str]:
    """facet.id -> stable display id ("F3") for every factor that ever entered
    the pool: numbered in the order they entered, never reused, so F3 means the
    same factor for the whole run (a replaced factor keeps its id in HISTORY)."""
    ids: Dict[str, str] = {}
    for r in state.ledger:
        if r.status in IN_POOL_STATUSES and r.facet_id and r.facet_id not in ids:
            ids[r.facet_id] = f"F{len(ids) + 1}"
    for facet in state.pool:  # a pool that did not come from the ledger
        ids.setdefault(facet.id, f"F{len(ids) + 1}")
    return ids


def _named(facet_id: str, titles: Dict[str, str], ids: Dict[str, str]) -> str:
    title = f"\"{titles.get(facet_id, '?')}\""
    return f"{ids[facet_id]} ({title})" if facet_id in ids else title


def _outcome(r: CandidateRecord, state: RunState, titles: Dict[str, str],
             ids: Dict[str, str]) -> str:
    s = r.status
    if s in IN_POOL_STATUSES:
        by = next((x.facet_id for x in state.ledger if x.replaced_facet_id == r.facet_id), None)
        return (f"was {ids.get(r.facet_id, 'in the pool')}; replaced in the pool by "
                + (_named(by, titles, ids) if by else "a later factor"))
    if s == CandidateStatus.DUPLICATE:
        return f"duplicate of {_named(r.duplicate_of, titles, ids)}"
    if s == CandidateStatus.IRR_PRUNED:
        return "unreliable: independent labelers disagreed too often"
    if s == CandidateStatus.VOCABULARY_FAILED:
        return "no usable values could be built from the tutorials"
    if s == CandidateStatus.LABELING_FAILED:
        return "the tutorial segments could not be labeled"
    if s == CandidateStatus.SCHEMA_REJECTED:
        return ("the pool was full and replacing any factor with it did not improve "
                "the design space")
    return s.value


def history_entry(r: CandidateRecord, state: RunState) -> HistoryEntry:
    """The HISTORY entry for one ledger record."""
    titles = {x.facet_id: x.title for x in state.ledger if x.facet_id and x.title}
    return HistoryEntry(title=r.title or "", scope=r.scope or "",
                        definition=r.definition or "",
                        outcome=_outcome(r, state, titles, factor_ids(state)))


def history_entries(state: RunState) -> List[HistoryEntry]:
    """Every tried factor that is not in the pool, in ledger order."""
    pool_ids = set(state.pool_ids())
    return [history_entry(r, state) for r in state.ledger
            if r.status not in NO_HISTORY_STATUSES and r.facet_id not in pool_ids]
