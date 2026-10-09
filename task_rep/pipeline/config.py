"""
Configuration for the prompt-based pipeline (a0 batch, a1/a2 iterative).

Shared models, budgets, lifecycle, evaluation, and sampling settings come from
``contextual.settings.RunSettings``; this module adds the proposal strategy,
the validated set of iterative context components, and ITEMS sampling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, Iterable

from contextual.settings import RunSettings


class ProposalStrategy(str, Enum):
    """How candidates are proposed."""

    BATCH = "batch"          # a0: one request for the whole budget
    ITERATIVE = "iterative"  # a1 / a2: one candidate per iteration


class ContextComponent(str, Enum):
    """One section of the iterative proposal prompt. Ignored for BATCH."""

    FACETS = "facets"    # current pool: definitions, vocabularies, label frequencies
    ITEMS = "items"      # tutorial segments with current pool labels
    SCHEMAS = "schemas"  # design space (DAG) of factor combinations + last submission
    HISTORY = "history"  # prior non-accepted candidates and their outcomes


A1_CONTEXT: FrozenSet[ContextComponent] = frozenset(
    {ContextComponent.FACETS, ContextComponent.ITEMS, ContextComponent.HISTORY}
)
A2_CONTEXT: FrozenSet[ContextComponent] = frozenset(
    {ContextComponent.SCHEMAS, ContextComponent.HISTORY}
)


def parse_context(components: Iterable[Any]) -> FrozenSet[ContextComponent]:
    """Validate and normalize a context component collection."""
    if isinstance(components, str):
        raise ValueError(
            f"context must be a list of components, not a string: {components!r}."
        )
    parsed = set()
    for c in components:
        try:
            parsed.add(ContextComponent(c))
        except ValueError:
            valid = [m.value for m in ContextComponent]
            raise ValueError(f"Unknown context component {c!r}; expected one of {valid}.")
    if not parsed:
        raise ValueError("context must contain at least one component.")
    return frozenset(parsed)


@dataclass
class PipelineConfig(RunSettings):
    """Full configuration for one pipeline run."""

    # ---- strategy ----
    strategy: ProposalStrategy = ProposalStrategy.ITERATIVE
    context: FrozenSet[ContextComponent] = field(default_factory=lambda: A1_CONTEXT)

    # ---- proposal context ----
    items_sample_videos: int = 0  # ITEMS context: videos per attempt (0 = all)

    def __post_init__(self):
        if not isinstance(self.strategy, ProposalStrategy):
            self.strategy = ProposalStrategy(self.strategy)
        self.context = parse_context(self.context)
        super().__post_init__()

    def semantic_dict(self) -> Dict[str, Any]:
        data = super().semantic_dict()
        if self.strategy == ProposalStrategy.BATCH:
            data.pop("context", None)  # a0 never uses context components
        return data

    def validate(self) -> None:
        super().validate()
        if self.items_sample_videos < 0:
            raise ValueError("items_sample_videos must be >= 0.")
