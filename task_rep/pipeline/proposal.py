"""
Facet proposal: protocol, typed contexts, and a VLM implementation.

The runner depends only on the ``FacetProposer`` protocol, never on an LLM
directly, so refusals and failures can be simulated deterministically in tests
and offline runs. a1 and a2 share one iterative implementation; only the
context component set differs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Protocol, Tuple, runtime_checkable

from utils import ModelConfig

from knowledge_items import KnowledgeItem, KnowledgeBase

from facets import Facet, AnnotationTable, VLMFacetExtractor

from optimization import AcceptedSchemaSummary, RejectedSchemaSummary

from contextual.feedback import HistoryEntry

from .config import ContextComponent


# ---------------------------------------------------------------------------
# Typed proposal contexts
# ---------------------------------------------------------------------------

@dataclass
class BatchContext:
    """Context for a0's single batch request."""

    task: str
    videos: Dict[str, List[KnowledgeItem]]  # the corpus (evaluated segments)
    num_facets: int
    seed: Optional[int] = None


@dataclass
class IterativeContext:
    """Context for a single a1/a2 proposal.

    Carries everything the requested ``components`` need; fields for components
    not in the set are left empty by the runner.
    """

    task: str
    components: FrozenSet[ContextComponent]
    corpus: Dict[str, List[KnowledgeItem]]  # every evaluated segment, by video (system message)
    pool: List[Facet] = field(default_factory=list)
    ids: Dict[str, str] = field(default_factory=dict)  # facet id -> stable "F3"
    annotations: AnnotationTable = field(default_factory=AnnotationTable)
    knowledge_base: Optional[KnowledgeBase] = None  # evaluated segments
    item_video_ids: Optional[List[str]] = None  # ITEMS videos (None = all)
    accepted_schemas: List[AcceptedSchemaSummary] = field(default_factory=list)
    rejected_schemas: List[RejectedSchemaSummary] = field(default_factory=list)
    # (source_id, target_id) refinement edges among accepted schemas
    refinement_edges: List[Tuple[str, str]] = field(default_factory=list)
    refinement_threshold: Optional[float] = None
    max_schema_facets: Optional[int] = None  # explained with SCHEMAS
    history: List[HistoryEntry] = field(default_factory=list)
    seed: Optional[int] = None  # distinct per attempt (see contextual.settings.stage_seed)

    def request_args(self) -> Dict[str, Any]:
        """Arguments of ``form_facet_iterative_request`` for this context."""
        return dict(
            task=self.task, facets=self.pool, components={c.value for c in self.components},
            corpus=self.corpus, item_video_ids=self.item_video_ids,
            annotations=self.annotations, knowledge_base=self.knowledge_base,
            accepted_schemas=self.accepted_schemas, rejected_schemas=self.rejected_schemas,
            refinement_edges=self.refinement_edges,
            refinement_threshold=self.refinement_threshold,
            max_schema_facets=self.max_schema_facets, history=self.history, ids=self.ids,
        )


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
        return self._extractor.propose_facet_iterative(seed=context.seed, **context.request_args())
