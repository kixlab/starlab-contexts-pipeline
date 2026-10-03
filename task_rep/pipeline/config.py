"""
Typed configuration for the contextual-unit pipeline.

Explicit enums, a validated set of context components, and independent
stopping controls (budget, attempts, refusal and no-improvement patience).
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any, Dict, FrozenSet, Iterable, Optional

from utils import ModelConfig
from utils.llm import SEED

from knowledge_items import ItemScopeConfig


class ProposalStrategy(str, Enum):
    """How candidates are proposed."""

    BATCH = "batch"          # a0: one request for the whole budget
    ITERATIVE = "iterative"  # a1 / a2: one candidate per iteration


class ContextComponent(str, Enum):
    """One section of the iterative proposal prompt. Ignored for BATCH."""

    FACETS = "facets"    # current pool: definitions, vocabularies, label frequencies
    ITEMS = "items"      # tutorial segments with current pool labels
    SCHEMAS = "schemas"  # accepted/rejected schema combinations with metrics
    HISTORY = "history"  # prior non-accepted candidates and their outcomes


A1_CONTEXT: FrozenSet[ContextComponent] = frozenset(
    {ContextComponent.FACETS, ContextComponent.ITEMS}
)
A2_CONTEXT: FrozenSet[ContextComponent] = frozenset({ContextComponent.SCHEMAS})


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


# Per-stage offsets keep the seeds of different call kinds apart.
_STAGE_SEED_OFFSETS = {
    "proposal": 0, "vocabulary": 100_000, "labeling": 200_000, "irr": 300_000,
    "items": 400_000,  # ITEMS video sampling
}


def stage_seed(run_seed: int, stage: str, index: int = 0) -> int:
    """Sampling seed for one call kind of one attempt in one run.

    Distinct per run seed, stage, and attempt index, so an iterative run never
    resends an identical prompt with an identical seed, and replicate runs
    (run seeds 0, 1, 2, ...) sample independently.
    """
    return SEED + run_seed * 1_000_000 + _STAGE_SEED_OFFSETS[stage] + index


# Settings that do not change what a run computes; excluded from the fingerprint.
NON_SEMANTIC_FIELDS = frozenset({"use_cache", "max_concurrency"})


@dataclass
class PipelineConfig:
    """Full configuration for one pipeline run."""

    # ---- per-stage models ----
    proposal_model: ModelConfig
    vocab_model: ModelConfig
    labeling_model: ModelConfig
    # Candidate dedup reuses the constructor's clustering embedder; the experiment
    # adapter fills this from the constructor's cluster_config so there is no
    # separate dedup-model knob to keep in sync.
    embedding_model: ModelConfig

    # ---- item scope ----
    item_scope: ItemScopeConfig

    # ---- strategy ----
    strategy: ProposalStrategy = ProposalStrategy.ITERATIVE
    context: FrozenSet[ContextComponent] = field(default_factory=lambda: A1_CONTEXT)

    # ---- budget / stopping (independent controls) ----
    proposal_budget: int = 30            # max unique candidates entering the lifecycle
    pool_size: int = 30                  # ceiling on selected facets (not a guarantee)
    attempt_limit: Optional[int] = None  # hard bound on attempts; None -> 3 x budget
    no_improvement_patience: int = 5     # consecutive processed candidates without gain
    refusal_patience: int = 5            # consecutive empty responses before stopping

    # ---- candidate lifecycle ----
    max_concurrency: int = 150
    vocab_sample_videos: int = 0            # 0 = all videos
    items_sample_videos: int = 0            # ITEMS context: videos per attempt (0 = all)
    dedup_similarity_threshold: float = 0.87
    irr_sample_videos: int = 3              # 0 = IRR disabled
    irr_num_raters: int = 3
    irr_min_agreement: float = 0.8

    # ---- schema evaluation (affects evaluation only, never stopping) ----
    metric_mode: str = "entropy"
    eta_squared_weighting: str = "equal"
    refinement_threshold: float = 0.2
    max_schema_facets: int = 1

    # ---- sampling ----
    seed: int = 0  # run seed: replicates of a config differ only here

    # ---- caching ----
    use_cache: bool = True

    def __post_init__(self):
        if not isinstance(self.strategy, ProposalStrategy):
            self.strategy = ProposalStrategy(self.strategy)
        self.context = parse_context(self.context)
        if self.attempt_limit is None:
            self.attempt_limit = 3 * self.proposal_budget
        self.validate()

    def validate(self) -> None:
        if self.proposal_budget <= 0:
            raise ValueError("proposal_budget must be > 0.")
        if self.pool_size <= 0:
            raise ValueError("pool_size must be > 0.")
        if self.pool_size > self.proposal_budget:
            raise ValueError(
                f"pool_size ({self.pool_size}) cannot exceed proposal_budget "
                f"({self.proposal_budget})."
            )
        if self.attempt_limit <= 0:
            raise ValueError("attempt_limit must be > 0.")
        if self.no_improvement_patience <= 0 or self.refusal_patience <= 0:
            raise ValueError("patience values must be > 0.")
        if self.seed < 0:
            raise ValueError("seed must be >= 0.")
        if self.vocab_sample_videos < 0 or self.items_sample_videos < 0:
            raise ValueError("sample_videos settings must be >= 0.")
        if not 0.0 < self.dedup_similarity_threshold <= 1.0:
            raise ValueError("dedup_similarity_threshold must be in (0, 1].")
        if self.metric_mode not in ("entropy", "coherence_ordinal", "coherence_temporal"):
            raise ValueError(f"Unknown metric_mode: {self.metric_mode!r}.")

    # ---- serialization ----

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal_model": self.proposal_model.to_dict(),
            "vocab_model": self.vocab_model.to_dict(),
            "labeling_model": self.labeling_model.to_dict(),
            "embedding_model": self.embedding_model.to_dict(),
            "item_scope": self.item_scope.to_dict(),
            "strategy": self.strategy.value,
            "context": sorted(c.value for c in self.context),
            "proposal_budget": self.proposal_budget,
            "pool_size": self.pool_size,
            "attempt_limit": self.attempt_limit,
            "no_improvement_patience": self.no_improvement_patience,
            "refusal_patience": self.refusal_patience,
            "max_concurrency": self.max_concurrency,
            "vocab_sample_videos": self.vocab_sample_videos,
            "items_sample_videos": self.items_sample_videos,
            "dedup_similarity_threshold": self.dedup_similarity_threshold,
            "irr_sample_videos": self.irr_sample_videos,
            "irr_num_raters": self.irr_num_raters,
            "irr_min_agreement": self.irr_min_agreement,
            "metric_mode": self.metric_mode,
            "eta_squared_weighting": self.eta_squared_weighting,
            "refinement_threshold": self.refinement_threshold,
            "max_schema_facets": self.max_schema_facets,
            "seed": self.seed,
            "use_cache": self.use_cache,
        }

    def semantic_dict(self) -> Dict[str, Any]:
        """``to_dict`` without settings that cannot change a run's outcome."""
        return {k: v for k, v in self.to_dict().items() if k not in NON_SEMANTIC_FIELDS}

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "PipelineConfig":
        known = {f.name for f in fields(PipelineConfig)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"Unknown pipeline config keys: {unknown}.")

        kwargs = dict(data)
        for key in ("proposal_model", "vocab_model", "labeling_model", "embedding_model"):
            kwargs[key] = ModelConfig.from_dict(data[key])
        kwargs["item_scope"] = ItemScopeConfig.from_dict(data["item_scope"])
        if "strategy" in data:
            kwargs["strategy"] = ProposalStrategy(data["strategy"])
        return PipelineConfig(**kwargs)
