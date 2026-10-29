"""
Deterministic fake proposer for offline pipeline runs and tests.

``ReplayProposer`` satisfies ``FacetProposer`` by replaying facets from a cached
run; the lifecycle fakes (vocabulary, labeler, embedder) live in
``contextual.fakes``.
"""

from __future__ import annotations

from typing import List, Optional, Set

from facets import Facet

from .proposal import BatchContext, IterativeContext


# ---------------------------------------------------------------------------
# Proposer
# ---------------------------------------------------------------------------

class ReplayProposer:
    """Replays corpus facets as proposals.

    Batch (a0): returns the first ``num_facets`` corpus facets.
    Iterative (a1/a2): returns one facet per call in corpus order, [] when
    exhausted. ``refuse_at`` forces a refusal ([]) at the given 0-based call
    indices without consuming a facet, to exercise refusal handling.
    """

    def __init__(self, corpus: List[Facet], refuse_at: Optional[Set[int]] = None):
        self.corpus = corpus
        self.refuse_at = refuse_at or set()
        self._cursor = 0
        self._calls = 0

    def propose_batch(self, context: BatchContext) -> List[Facet]:
        return self.corpus[: context.num_facets]

    def propose_one(self, context: IterativeContext) -> List[Facet]:
        call = self._calls
        self._calls += 1
        if call in self.refuse_at:
            return []
        if self._cursor >= len(self.corpus):
            return []
        facet = self.corpus[self._cursor]
        self._cursor += 1
        return [facet]

    def prompt_fingerprint(self) -> str:
        return "fake-replay-v1"
