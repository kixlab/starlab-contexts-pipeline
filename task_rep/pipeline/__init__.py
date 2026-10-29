"""
Contextual-unit construction pipeline.

Supports a0 (batch), a1 (iterative facets+items), and a2 (iterative schema
feedback) through one runner. The runner depends only on injectable services
(proposer, vocabulary builder, labeler, embedder) so runs can be simulated
offline with the fakes in ``pipeline.fakes``.
"""

from .config import (
    PipelineConfig,
    ProposalStrategy,
    ContextComponent,
    A1_CONTEXT,
    A2_CONTEXT,
)
from .models import (
    PipelineState,
    PipelineResult,
    CandidateRecord,
    CandidateStatus,
    StopReason,
    TreeQualitySnapshot,
)
from .proposal import (
    FacetProposer,
    VLMFacetProposer,
    BatchContext,
    IterativeContext,
    HistoryEntry,
)
from .candidates import (
    CandidateProcessor,
    ProcessedCandidate,
    CandidateFailure,
    Embedder,
    VocabularyBuilder,
    Labeler,
    dedup_text,
)
from .evaluation import PoolEvaluator, SelectionDecision
from .store import PipelineStore, kb_fingerprint
from .fingerprint import prompt_fingerprint
from .runner import PipelineRunner

__all__ = [
    "PipelineConfig",
    "ProposalStrategy",
    "ContextComponent",
    "A1_CONTEXT",
    "A2_CONTEXT",
    "PipelineState",
    "PipelineResult",
    "CandidateRecord",
    "CandidateStatus",
    "StopReason",
    "TreeQualitySnapshot",
    "FacetProposer",
    "VLMFacetProposer",
    "BatchContext",
    "IterativeContext",
    "HistoryEntry",
    "CandidateProcessor",
    "ProcessedCandidate",
    "CandidateFailure",
    "Embedder",
    "VocabularyBuilder",
    "Labeler",
    "dedup_text",
    "PoolEvaluator",
    "SelectionDecision",
    "PipelineStore",
    "kb_fingerprint",
    "prompt_fingerprint",
    "PipelineRunner",
]
