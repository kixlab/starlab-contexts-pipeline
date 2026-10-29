"""
Schema evaluation and pool selection.

Wraps the existing pure tree/schema functions. Owns the pool-update algorithm:
unconditional add below ``pool_size``; strict-improvement replacement at
capacity. Thresholds affect schema evaluation only, never stopping.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

from knowledge_items import KnowledgeBase

from facets import Facet, AnnotationTable

from optimization import (
    SchemaEvaluation,
    build_tree_nodes,
    evaluate_schemas,
    compare_tree_quality,
)

from .settings import RunSettings
from .lifecycle import ProcessedCandidate
from .models import CandidateStatus, RunState, TreeQualitySnapshot

logger = logging.getLogger(__name__)


@dataclass
class SelectionDecision:
    """Outcome of considering one processed candidate for the pool."""

    status: CandidateStatus  # ACCEPTED | REPLACED | SCHEMA_REJECTED
    improved: bool
    snapshot: TreeQualitySnapshot
    replaced_facet_id: Optional[str] = None


def _snapshot(state: RunState) -> TreeQualitySnapshot:
    tree = state.tree
    nodes = len(tree.nodes) if tree else 0
    unique = len({fid for n in (tree.nodes if tree else []) for fid in n.facet_ids})
    return TreeQualitySnapshot(
        tree_nodes=nodes, unique_facets_in_tree=unique, pool_ids=state.pool_ids(),
    )


class PoolEvaluator:
    """Builds schemas and decides pool membership for a focused knowledge base."""

    def __init__(self, config: RunSettings, focused_kb: KnowledgeBase):
        self.config = config
        self.focused_kb = focused_kb

    def evaluate_pool(
        self, facets: List[Facet], annotations: AnnotationTable,
    ) -> SchemaEvaluation:
        """Full schema evaluation for a pool."""
        nodes, h_memo = build_tree_nodes(
            self.focused_kb, facets, annotations,
            max_schema_facets=self.config.max_schema_facets,
            metric_mode=self.config.metric_mode,
            eta_squared_weighting=self.config.eta_squared_weighting,
        )
        return evaluate_schemas(
            nodes, self.focused_kb, {f.id: f for f in facets}, annotations, h_memo,
            refinement_threshold=self.config.refinement_threshold,
            metric_mode=self.config.metric_mode,
        )

    def apply_evaluation(self, state: RunState, evaluation: SchemaEvaluation):
        state.tree = evaluation.tree
        state.accepted_schemas = evaluation.accepted_schemas
        state.rejected_schemas = evaluation.rejected_schemas

    def initialize(self, state: RunState):
        """Compute the schema evaluation for the current (possibly empty) pool."""
        if state.pool:
            self.apply_evaluation(
                state, self.evaluate_pool(state.pool, state.annotations)
            )

    def consider(
        self, state: RunState, candidate: ProcessedCandidate,
    ) -> SelectionDecision:
        """Try to place ``candidate`` into the pool, mutating ``state`` on success.

        Below ``pool_size``: add unconditionally (always an improvement).
        At capacity: replace the single member that yields the best strictly
        improving tree; otherwise reject and leave the pool unchanged.
        """
        pool = list(state.pool)
        merged_ann = state.annotations.merge(candidate.annotations)

        if len(pool) < self.config.pool_size:
            pool.append(candidate.facet)
            state.pool = pool
            state.annotations = merged_ann.filter_facets(state.pool_ids())
            self.apply_evaluation(
                state, self.evaluate_pool(state.pool, state.annotations)
            )
            logger.info("Added facet %s to pool (%d/%d).",
                        candidate.facet.id, len(pool), self.config.pool_size)
            return SelectionDecision(
                status=CandidateStatus.ACCEPTED, improved=True,
                snapshot=_snapshot(state),
            )

        # At capacity: union-precompute, then filter per replacement trial.
        union_pool = pool + [candidate.facet]
        union_ann = merged_ann.filter_facets([f.id for f in union_pool])
        all_nodes, h_memo = build_tree_nodes(
            self.focused_kb, union_pool, union_ann,
            max_schema_facets=self.config.max_schema_facets,
            metric_mode=self.config.metric_mode,
            eta_squared_weighting=self.config.eta_squared_weighting,
        )
        union_by_id = {f.id: f for f in union_pool}

        best_tree = state.tree
        best_eval: Optional[SchemaEvaluation] = None
        best_idx = -1
        for i in range(len(pool)):
            trial_ids = {f.id for f in pool[:i] + [candidate.facet] + pool[i + 1:]}
            trial_nodes = [n for n in all_nodes if set(n.facet_ids) <= trial_ids]
            trial_eval = evaluate_schemas(
                trial_nodes, self.focused_kb, union_by_id, union_ann, h_memo,
                refinement_threshold=self.config.refinement_threshold,
                metric_mode=self.config.metric_mode,
            )
            if compare_tree_quality(trial_eval.tree, best_tree) > 0:
                best_tree = trial_eval.tree
                best_eval = trial_eval
                best_idx = i

        if best_idx >= 0 and best_eval is not None:
            replaced = pool[best_idx]
            pool[best_idx] = candidate.facet
            state.pool = pool
            state.annotations = merged_ann.filter_facets(state.pool_ids())
            self.apply_evaluation(state, best_eval)
            logger.info("Replaced facet %s with %s (tree: %d nodes).",
                        replaced.id, candidate.facet.id, len(best_tree.nodes))
            return SelectionDecision(
                status=CandidateStatus.REPLACED, improved=True,
                snapshot=_snapshot(state), replaced_facet_id=replaced.id,
            )

        return SelectionDecision(
            status=CandidateStatus.SCHEMA_REJECTED, improved=False,
            snapshot=_snapshot(state),
        )
