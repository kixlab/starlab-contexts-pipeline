"""
Settings shared by every contextual-unit runner (pipeline and agentic).

``RunSettings`` holds the models, item scope, budget/stopping controls,
candidate-lifecycle, schema-evaluation, and sampling settings. Runner-specific
configs subclass it and add their own fields; serialization is generic over
dataclass fields, so subclasses only convert their own field types.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
from typing import Any, Dict, Optional

from utils import ModelConfig
from utils.llm import SEED

from knowledge_items import ItemScopeConfig


# Per-stage offsets keep the seeds of different call kinds apart.
_STAGE_SEED_OFFSETS = {
    "proposal": 0, "vocabulary": 100_000, "labeling": 200_000, "irr": 300_000,
    "items": 400_000,  # ITEMS video sampling (pipeline)
    "agent": 500_000,  # agent turns (agentic)
}


def stage_seed(run_seed: int, stage: str, index: int = 0) -> int:
    """Sampling seed for one call kind of one attempt in one run.

    Distinct per run seed, stage, and attempt index, so a runner never resends
    an identical prompt with an identical seed, and replicate runs (run seeds
    0, 1, 2, ...) sample independently.
    """
    return SEED + run_seed * 1_000_000 + _STAGE_SEED_OFFSETS[stage] + index


# Settings that do not change what a run computes; excluded from the fingerprint.
NON_SEMANTIC_FIELDS = frozenset({"use_cache", "max_concurrency"})

_MODEL_FIELDS = ("proposal_model", "vocab_model", "labeling_model", "embedding_model")


@dataclass
class RunSettings:
    """Settings every runner shares."""

    # ---- per-stage models ----
    proposal_model: ModelConfig   # proposes candidates (prompt or agent)
    vocab_model: ModelConfig
    labeling_model: ModelConfig
    # Candidate dedup reuses the constructor's clustering embedder; experiment
    # adapters fill this from the constructor's cluster_config.
    embedding_model: ModelConfig

    # ---- item scope ----
    item_scope: ItemScopeConfig

    # ---- corpus (see contextual.corpus) ----
    corpus_sample_videos: int = 0  # restrict the KB to this many videos (0 = all)
    corpus_seed: int = 0           # fixes the sample; independent of the run seed

    # ---- budget / stopping (independent controls) ----
    proposal_budget: int = 30            # max unique candidates entering the lifecycle
    pool_size: int = 30                  # ceiling on selected facets (not a guarantee)
    attempt_limit: Optional[int] = None  # hard bound on attempts; None -> 3 x budget
    no_improvement_patience: int = 5     # consecutive processed candidates without gain
    refusal_patience: int = 5            # consecutive empty responses before stopping

    # ---- candidate lifecycle ----
    max_concurrency: int = 150
    vocab_sample_videos: int = 0            # 0 = all videos
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
        if self.vocab_sample_videos < 0 or self.corpus_sample_videos < 0:
            raise ValueError("sample_videos settings must be >= 0.")
        if not 0.0 < self.dedup_similarity_threshold <= 1.0:
            raise ValueError("dedup_similarity_threshold must be in (0, 1].")
        if self.metric_mode not in ("entropy", "coherence_ordinal", "coherence_temporal"):
            raise ValueError(f"Unknown metric_mode: {self.metric_mode!r}.")

    # ---- serialization ----

    def to_dict(self) -> Dict[str, Any]:
        return {f.name: _to_json(getattr(self, f.name)) for f in fields(self)}

    def semantic_dict(self) -> Dict[str, Any]:
        """``to_dict`` without settings that cannot change a run's outcome."""
        return {k: v for k, v in self.to_dict().items() if k not in NON_SEMANTIC_FIELDS}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"Unknown {cls.__name__} keys: {unknown}.")
        kwargs = dict(data)
        for key in _MODEL_FIELDS:
            kwargs[key] = ModelConfig.from_dict(data[key])
        kwargs["item_scope"] = ItemScopeConfig.from_dict(data["item_scope"])
        return cls(**kwargs)


def _to_json(value: Any) -> Any:
    if isinstance(value, (ModelConfig, ItemScopeConfig)):
        return value.to_dict()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, frozenset):
        return sorted(v.value if isinstance(v, Enum) else v for v in value)
    return value
