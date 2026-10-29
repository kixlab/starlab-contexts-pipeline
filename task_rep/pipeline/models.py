"""
Pipeline-level state and result models.

Reuses ``Facet`` / ``AnnotationTable`` from ``facets`` and the schema summaries
from ``optimization`` rather than duplicating domain models. Everything here is
serializable so the runner can checkpoint after every iteration and the CLI can
replay a saved run's candidate ledger.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from facets import Facet, AnnotationTable

from optimization import (
    RefinementTree,
    AcceptedSchemaSummary,
    RejectedSchemaSummary,
)


class CandidateStatus(str, Enum):
    """Terminal outcome recorded for every candidate that enters processing."""

    REFUSED = "refused"                  # empty model response
    INVALID = "invalid"                  # unparseable / structurally invalid
    DUPLICATE = "duplicate"              # too similar to a prior candidate
    IRR_PRUNED = "irr_pruned"
    VOCABULARY_FAILED = "vocabulary_failed"
    LABELING_FAILED = "labeling_failed"
    SCHEMA_REJECTED = "schema_rejected"  # processed but did not improve the pool
    ACCEPTED = "accepted"                # added to a non-full pool
    REPLACED = "replaced"                # replaced a member of a full pool
    TRUNCATED = "truncated"              # a0 only: beyond proposal_budget, not processed


class StopReason(str, Enum):
    """Why the run loop terminated.

    Iterative bounds are listed in precedence order: when several trip on the
    same step, the earliest listed wins.
    """

    PROPOSAL_BUDGET = "proposal_budget"
    ATTEMPT_LIMIT = "attempt_limit"
    REFUSAL_PATIENCE = "refusal_patience"
    NO_IMPROVEMENT_PATIENCE = "no_improvement_patience"
    BATCH_COMPLETE = "batch_complete"  # a0 processed its single batch
    BATCH_REFUSED = "batch_refused"    # a0 batch response was empty


@dataclass
class TreeQualitySnapshot:
    """Pool tree quality captured immediately after a selection decision.

    This is what the ledger-based budget analysis replays.
    """

    tree_nodes: int
    unique_facets_in_tree: int
    pool_ids: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tree_nodes": self.tree_nodes,
            "unique_facets_in_tree": self.unique_facets_in_tree,
            "pool_ids": list(self.pool_ids),
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "TreeQualitySnapshot":
        return TreeQualitySnapshot(
            tree_nodes=data["tree_nodes"],
            unique_facets_in_tree=data["unique_facets_in_tree"],
            pool_ids=list(data.get("pool_ids", [])),
        )


@dataclass
class CandidateRecord:
    """One entry in the candidate ledger — every proposal ends here."""

    attempt_index: int                       # 1-based attempt counter value
    status: CandidateStatus
    facet_id: Optional[str] = None
    title: Optional[str] = None
    scope: Optional[str] = None
    definition: Optional[str] = None
    counts_against_budget: bool = False      # entered the lifecycle (a "proposal")
    # dedup diagnostics: nearest prior lifecycle candidate, for every embedded
    # candidate (so the threshold can be tuned from saved ledgers)
    nearest_id: Optional[str] = None
    nearest_similarity: Optional[float] = None
    duplicate_of: Optional[str] = None
    # selection diagnostics
    replaced_facet_id: Optional[str] = None
    tree_quality: Optional[TreeQualitySnapshot] = None
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attempt_index": self.attempt_index,
            "status": self.status.value,
            "facet_id": self.facet_id,
            "title": self.title,
            "scope": self.scope,
            "definition": self.definition,
            "counts_against_budget": self.counts_against_budget,
            "nearest_id": self.nearest_id,
            "nearest_similarity": self.nearest_similarity,
            "duplicate_of": self.duplicate_of,
            "replaced_facet_id": self.replaced_facet_id,
            "tree_quality": self.tree_quality.to_dict() if self.tree_quality else None,
            "note": self.note,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "CandidateRecord":
        tq = data.get("tree_quality")
        return CandidateRecord(
            attempt_index=data["attempt_index"],
            status=CandidateStatus(data["status"]),
            facet_id=data.get("facet_id"),
            title=data.get("title"),
            scope=data.get("scope"),
            definition=data.get("definition"),
            counts_against_budget=data.get("counts_against_budget", False),
            nearest_id=data.get("nearest_id"),
            nearest_similarity=data.get("nearest_similarity"),
            duplicate_of=data.get("duplicate_of"),
            replaced_facet_id=data.get("replaced_facet_id"),
            tree_quality=TreeQualitySnapshot.from_dict(tq) if tq else None,
            note=data.get("note", ""),
        )


@dataclass
class PipelineState:
    """Mutable state carried through a run and checkpointed each iteration."""

    # selected pool, ordered by insertion (deterministic for tie-breaks/checkpoints)
    pool: List[Facet] = field(default_factory=list)
    annotations: AnnotationTable = field(default_factory=AnnotationTable)
    tree: Optional[RefinementTree] = None
    accepted_schemas: List[AcceptedSchemaSummary] = field(default_factory=list)
    rejected_schemas: List[RejectedSchemaSummary] = field(default_factory=list)

    # complete candidate ledger
    ledger: List[CandidateRecord] = field(default_factory=list)

    # dedup embeddings (facet_id -> vector) of candidates that entered the
    # lifecycle, persisted so dedup survives checkpointing; duplicates, refusals,
    # and invalid candidates are never added (no A->B->C chaining)
    embeddings: Dict[str, List[float]] = field(default_factory=dict)

    # counters
    attempts: int = 0
    proposals: int = 0
    refusal_streak: int = 0
    no_improve_streak: int = 0
    iteration: int = 0

    # exact-run cache identity
    fingerprint: str = ""
    stop_reason: Optional[StopReason] = None

    def pool_ids(self) -> List[str]:
        return [f.id for f in self.pool]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pool": [f.to_dict() for f in self.pool],
            "annotations": self.annotations.to_list(),
            "tree": (self.tree or RefinementTree(nodes=[], edges=[])).to_dict(),
            "accepted_schemas": [s.to_dict() for s in self.accepted_schemas],
            "rejected_schemas": [s.to_dict() for s in self.rejected_schemas],
            "ledger": [r.to_dict() for r in self.ledger],
            "embeddings": self.embeddings,
            "attempts": self.attempts,
            "proposals": self.proposals,
            "refusal_streak": self.refusal_streak,
            "no_improve_streak": self.no_improve_streak,
            "iteration": self.iteration,
            "fingerprint": self.fingerprint,
            "stop_reason": self.stop_reason.value if self.stop_reason else None,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "PipelineState":
        stop = data.get("stop_reason")
        return PipelineState(
            pool=[Facet.from_dict(f) for f in data.get("pool", [])],
            annotations=AnnotationTable.from_list(data.get("annotations", [])),
            tree=RefinementTree.from_dict(data["tree"]) if data.get("tree") else None,
            accepted_schemas=[
                AcceptedSchemaSummary.from_dict(s)
                for s in data.get("accepted_schemas", [])
            ],
            rejected_schemas=[
                RejectedSchemaSummary.from_dict(s)
                for s in data.get("rejected_schemas", [])
            ],
            ledger=[CandidateRecord.from_dict(r) for r in data.get("ledger", [])],
            embeddings={k: list(v) for k, v in data.get("embeddings", {}).items()},
            attempts=data.get("attempts", 0),
            proposals=data.get("proposals", 0),
            refusal_streak=data.get("refusal_streak", 0),
            no_improve_streak=data.get("no_improve_streak", 0),
            iteration=data.get("iteration", 0),
            fingerprint=data.get("fingerprint", ""),
            stop_reason=StopReason(stop) if stop else None,
        )


@dataclass
class PipelineResult:
    """Serialized outcome of a run. Does not embed the knowledge base — only its
    fingerprint; the adapter reloads the KB from the constructor when needed.

    Self-describing: carries the semantic config, prompt fingerprint, and the
    focused KB's unit count, so saved results can be found, summarized, and
    compared without loading a knowledge base or calling a model.
    """

    pool: List[Facet]
    annotations: AnnotationTable
    tree: Optional[RefinementTree]
    accepted_schemas: List[AcceptedSchemaSummary]
    rejected_schemas: List[RejectedSchemaSummary]
    ledger: List[CandidateRecord]
    stop_reason: Optional[StopReason]
    attempts: int
    proposals: int
    iterations: int
    fingerprint: str
    kb_fingerprint: str = ""
    prompt_fingerprint: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    num_units: int = 0

    @staticmethod
    def from_state(
        state: PipelineState,
        kb_fingerprint: str = "",
        prompt_fingerprint: str = "",
        config: Optional[Dict[str, Any]] = None,
        num_units: int = 0,
    ) -> "PipelineResult":
        return PipelineResult(
            pool=list(state.pool),
            annotations=state.annotations,
            tree=state.tree,
            accepted_schemas=list(state.accepted_schemas),
            rejected_schemas=list(state.rejected_schemas),
            ledger=list(state.ledger),
            stop_reason=state.stop_reason,
            attempts=state.attempts,
            proposals=state.proposals,
            iterations=state.iteration,
            fingerprint=state.fingerprint,
            kb_fingerprint=kb_fingerprint,
            prompt_fingerprint=prompt_fingerprint,
            config=dict(config or {}),
            num_units=num_units,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pool": [f.to_dict() for f in self.pool],
            "annotations": self.annotations.to_list(),
            "tree": (self.tree or RefinementTree(nodes=[], edges=[])).to_dict(),
            "accepted_schemas": [s.to_dict() for s in self.accepted_schemas],
            "rejected_schemas": [s.to_dict() for s in self.rejected_schemas],
            "ledger": [r.to_dict() for r in self.ledger],
            "stop_reason": self.stop_reason.value if self.stop_reason else None,
            "attempts": self.attempts,
            "proposals": self.proposals,
            "iterations": self.iterations,
            "fingerprint": self.fingerprint,
            "kb_fingerprint": self.kb_fingerprint,
            "prompt_fingerprint": self.prompt_fingerprint,
            "config": self.config,
            "num_units": self.num_units,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "PipelineResult":
        stop = data.get("stop_reason")
        return PipelineResult(
            pool=[Facet.from_dict(f) for f in data.get("pool", [])],
            annotations=AnnotationTable.from_list(data.get("annotations", [])),
            tree=RefinementTree.from_dict(data["tree"]) if data.get("tree") else None,
            accepted_schemas=[
                AcceptedSchemaSummary.from_dict(s)
                for s in data.get("accepted_schemas", [])
            ],
            rejected_schemas=[
                RejectedSchemaSummary.from_dict(s)
                for s in data.get("rejected_schemas", [])
            ],
            ledger=[CandidateRecord.from_dict(r) for r in data.get("ledger", [])],
            stop_reason=StopReason(stop) if stop else None,
            attempts=data.get("attempts", 0),
            proposals=data.get("proposals", 0),
            iterations=data.get("iterations", 0),
            fingerprint=data.get("fingerprint", ""),
            kb_fingerprint=data.get("kb_fingerprint", ""),
            prompt_fingerprint=data.get("prompt_fingerprint", ""),
            config=data.get("config", {}),
            num_units=data.get("num_units", 0),
        )
