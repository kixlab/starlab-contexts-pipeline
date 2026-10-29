"""
Prompt-based contextual-unit construction: a0 (batch), a1 (iterative
facets+items), and a2 (iterative schema feedback) through one runner.

Builds on the shared core in ``contextual`` (candidate lifecycle, pool
evaluation, ledger, store); never imports the ``agentic`` package.
"""

from .config import (
    PipelineConfig,
    ProposalStrategy,
    ContextComponent,
    A1_CONTEXT,
    A2_CONTEXT,
)
from .proposal import (
    FacetProposer,
    VLMFacetProposer,
    BatchContext,
    IterativeContext,
    HistoryEntry,
)
from .fingerprint import prompt_fingerprint
from .runner import PipelineRunner

__all__ = [
    "PipelineConfig",
    "ProposalStrategy",
    "ContextComponent",
    "A1_CONTEXT",
    "A2_CONTEXT",
    "FacetProposer",
    "VLMFacetProposer",
    "BatchContext",
    "IterativeContext",
    "HistoryEntry",
    "prompt_fingerprint",
    "PipelineRunner",
]
