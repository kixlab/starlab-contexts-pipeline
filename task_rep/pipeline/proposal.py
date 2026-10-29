"""
Facet proposal: protocol, typed contexts, and a VLM implementation.

The runner depends only on the ``FacetProposer`` protocol, never on an LLM
directly, so refusals and failures can be simulated deterministically in tests
and offline runs. a1 and a2 share one iterative implementation; only the
context component set differs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Protocol, runtime_checkable

from utils import ModelConfig

from knowledge_items import KnowledgeItem, KnowledgeBase

from facets import Facet, AnnotationTable, VLMFacetExtractor

from optimization import AcceptedSchemaSummary, RejectedSchemaSummary

from .config import ContextComponent


# ---------------------------------------------------------------------------
# Typed proposal contexts
# ---------------------------------------------------------------------------

@dataclass
class BatchContext:
    """Context for a0's single batch request."""

    task: str
    videos: Dict[str, List[KnowledgeItem]]
    num_facets: int
    seed: Optional[int] = None


@dataclass
class HistoryEntry:
    """A previously tried candidate that is not in the pool (HISTORY component).

    ``outcome`` is the candidate's ledger status value, or ``"replaced_out"`` for
    a facet that was accepted and later evicted from the pool.
    """

    title: str
    scope: str
    definition: str
    outcome: str
    duplicate_of: Optional[str] = None  # title of the matched candidate


@dataclass
class IterativeContext:
    """Context for a single a1/a2 proposal.

    Carries everything the requested ``components`` need; fields for components
    not in the set are left empty by the runner.
    """

    task: str
    components: FrozenSet[ContextComponent]
    pool: List[Facet] = field(default_factory=list)
    annotations: AnnotationTable = field(default_factory=AnnotationTable)
    knowledge_base: Optional[KnowledgeBase] = None
    videos: Optional[Dict[str, List[KnowledgeItem]]] = None
    accepted_schemas: List[AcceptedSchemaSummary] = field(default_factory=list)
    rejected_schemas: List[RejectedSchemaSummary] = field(default_factory=list)
    history: List[HistoryEntry] = field(default_factory=list)
    seed: Optional[int] = None  # distinct per attempt (see config.stage_seed)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class FacetProposer(Protocol):
    """A source of facet candidates. Empty return = refusal."""

    def propose_batch(self, context: BatchContext) -> List[Facet]:
        """Propose up to ``context.num_facets`` facets in one shot (a0)."""
        ...

    def propose_one(self, context: IterativeContext) -> List[Facet]:
        """Propose one candidate (a1/a2). Return [] to signal a refusal."""
        ...


# ---------------------------------------------------------------------------
# VLM implementation
# ---------------------------------------------------------------------------

class VLMFacetProposer:
    """FacetProposer backed by the existing ``VLMFacetExtractor`` prompts."""

    def __init__(self, model_config: ModelConfig, max_concurrency: int = 32):
        self._extractor = VLMFacetExtractor(
            model_config=model_config,
            max_concurrency=max_concurrency,
        )

    def propose_batch(self, context: BatchContext) -> List[Facet]:
        return self._extractor.init_facets(
            task=context.task,
            videos=context.videos,
            num_facets=context.num_facets,
            seed=context.seed,
        )

    def propose_one(self, context: IterativeContext) -> List[Facet]:
        return self._extractor.propose_facet_iterative(
            task=context.task,
            facets=context.pool,
            components={c.value for c in context.components},
            videos=context.videos,
            annotations=context.annotations,
            knowledge_base=context.knowledge_base,
            accepted_schemas=context.accepted_schemas,
            rejected_schemas=context.rejected_schemas,
            history=context.history,
            seed=context.seed,
        )
