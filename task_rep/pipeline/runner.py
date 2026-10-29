"""
Pipeline orchestration.

Coordinates proposer, candidate lifecycle, and pool evaluation. Contains no
prompt construction, metrics, selection algorithm, or JSON serialization — those
live in their own modules. Consumes a ready ``KnowledgeBase``; extraction lives
in the outer experiment adapter.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import List, Optional

from knowledge_items import KnowledgeBase

from facets import Facet

from utils.llm import USAGE, usage_delta, usage_stage

from optimization import count_unique_units

from .config import PipelineConfig, ProposalStrategy, ContextComponent, stage_seed
from .candidates import (
    CandidateProcessor, ProcessedCandidate, CandidateFailure, CandidateOutcome,
    Embedder, VocabularyBuilder, Labeler,
)
from .evaluation import PoolEvaluator
from .fingerprint import prompt_fingerprint
from .models import (
    PipelineState, PipelineResult, CandidateRecord, CandidateStatus,
    StopReason, TreeQualitySnapshot,
)
from .proposal import FacetProposer, BatchContext, IterativeContext, HistoryEntry
from .store import PipelineStore, kb_fingerprint

logger = logging.getLogger(__name__)

# Statuses that mean the candidate entered the lifecycle (counts against budget).
# Their embeddings join the dedup set.
_BUDGET_STATUSES = {
    CandidateStatus.IRR_PRUNED,
    CandidateStatus.VOCABULARY_FAILED,
    CandidateStatus.LABELING_FAILED,
    CandidateStatus.SCHEMA_REJECTED,
    CandidateStatus.ACCEPTED,
    CandidateStatus.REPLACED,
}
# Statuses that increment the no-improvement streak (processed but no gain).
_NO_IMPROVE_STATUSES = {
    CandidateStatus.DUPLICATE,
    CandidateStatus.IRR_PRUNED,
    CandidateStatus.VOCABULARY_FAILED,
    CandidateStatus.LABELING_FAILED,
    CandidateStatus.SCHEMA_REJECTED,
}
# Statuses never shown in HISTORY (no meaningful candidate to avoid).
_NO_HISTORY_STATUSES = {
    CandidateStatus.REFUSED,
    CandidateStatus.INVALID,
    CandidateStatus.TRUNCATED,
}


class PipelineRunner:
    """Runs a0/a1/a2 for one knowledge base."""

    def __init__(
        self,
        config: PipelineConfig,
        task: str,
        knowledge_base: KnowledgeBase,
        proposer: FacetProposer,
        vocab_builder: VocabularyBuilder,
        labeler: Labeler,
        embedder: Embedder,
        cache_dir: str | Path,
    ):
        self.config = config
        self.task = task
        self.knowledge_base = knowledge_base

        # Focused KB for optimization scope (subtype filter); proposal context,
        # vocabulary, and labeling still see the full knowledge base.
        focused = knowledge_base
        subtype_filter = config.item_scope.atomic_subtype
        if subtype_filter.subtypes:
            focused = knowledge_base.filter_subtypes(subtype_filter.subtypes)
        self.focused_kb = focused

        self.proposer = proposer
        self.processor = CandidateProcessor(
            config=config, task=task, knowledge_base=knowledge_base,
            vocab_builder=vocab_builder, labeler=labeler, embedder=embedder,
        )
        self.evaluator = PoolEvaluator(config=config, focused_kb=focused)

        self.kb_fp = kb_fingerprint(knowledge_base)
        self.prompt_fp = prompt_fingerprint(config)
        self.store = PipelineStore(
            cache_dir=cache_dir, config=config,
            kb_fp=self.kb_fp, prompt_fp=self.prompt_fp,
        )

    # ---- ledger + counter bookkeeping (the normative counter table) ----

    @staticmethod
    def _snapshot(state: PipelineState) -> TreeQualitySnapshot:
        tree = state.tree
        nodes = len(tree.nodes) if tree else 0
        unique = len({fid for n in (tree.nodes if tree else []) for fid in n.facet_ids})
        return TreeQualitySnapshot(nodes, unique, state.pool_ids())

    def _record(
        self, state: PipelineState, status: CandidateStatus, *,
        increment_attempts: bool = True,
        facet: Optional[Facet] = None,
        nearest_id: Optional[str] = None,
        nearest_similarity: Optional[float] = None,
        duplicate_of: Optional[str] = None,
        replaced_facet_id: Optional[str] = None,
        snapshot: Optional[TreeQualitySnapshot] = None,
        note: str = "",
    ) -> None:
        if status == CandidateStatus.TRUNCATED:
            # a0 surplus: recorded for completeness, touches no counter.
            increment_attempts = False
        else:
            state.refusal_streak = (state.refusal_streak + 1
                                    if status == CandidateStatus.REFUSED else 0)
            if status in (CandidateStatus.ACCEPTED, CandidateStatus.REPLACED):
                state.no_improve_streak = 0
            elif status in _NO_IMPROVE_STATUSES:
                state.no_improve_streak += 1
            # REFUSED / INVALID leave no_improve_streak unchanged
        if increment_attempts:
            state.attempts += 1
        counts = status in _BUDGET_STATUSES
        if counts:
            state.proposals += 1

        state.ledger.append(CandidateRecord(
            attempt_index=state.attempts,
            status=status,
            facet_id=facet.id if facet else None,
            title=facet.title if facet else None,
            scope=facet.scope if facet else None,
            definition=facet.definition if facet else None,
            counts_against_budget=counts,
            nearest_id=nearest_id,
            nearest_similarity=nearest_similarity,
            duplicate_of=duplicate_of,
            replaced_facet_id=replaced_facet_id,
            tree_quality=snapshot if snapshot is not None else self._snapshot(state),
            note=note,
        ))

    def _stop_reason(self, state: PipelineState) -> Optional[StopReason]:
        """First tripped bound, in precedence order."""
        c = self.config
        if state.proposals >= c.proposal_budget:
            return StopReason.PROPOSAL_BUDGET
        if state.attempts >= c.attempt_limit:
            return StopReason.ATTEMPT_LIMIT
        if state.refusal_streak >= c.refusal_patience:
            return StopReason.REFUSAL_PATIENCE
        if state.no_improve_streak >= c.no_improvement_patience:
            return StopReason.NO_IMPROVEMENT_PATIENCE
        return None

    def _apply_outcome(self, state: PipelineState, outcome: CandidateOutcome,
                       *, increment_attempts: bool) -> None:
        """Run pool selection for a processed candidate, record any outcome."""
        if isinstance(outcome, ProcessedCandidate):
            state.embeddings[outcome.facet.id] = outcome.embedding
            decision = self.evaluator.consider(state, outcome)
            self._record(
                state, decision.status, increment_attempts=increment_attempts,
                facet=outcome.facet, nearest_id=outcome.nearest_id,
                nearest_similarity=outcome.nearest_similarity,
                replaced_facet_id=decision.replaced_facet_id,
                snapshot=decision.snapshot, note=outcome.note,
            )
            return

        failure: CandidateFailure = outcome
        if failure.status in _BUDGET_STATUSES and failure.embedding is not None:
            state.embeddings[failure.facet.id] = failure.embedding
        self._record(
            state, failure.status, increment_attempts=increment_attempts,
            facet=failure.facet, nearest_id=failure.nearest_id,
            nearest_similarity=failure.nearest_similarity,
            duplicate_of=failure.duplicate_of, note=failure.note,
        )

    # ---- public entry points ----

    def run(self) -> PipelineResult:
        state: Optional[PipelineState] = None
        if self.config.use_cache:
            state = self.store.restore_checkpoint()
        if state is None:
            state = PipelineState(fingerprint=self.store.fingerprint)

        if state.stop_reason is not None:
            logger.info("Run already complete (%s).", state.stop_reason.value)
            return self._result(state)

        if self.config.strategy == ProposalStrategy.BATCH:
            self._run_batch(state)
        else:
            self._run_iterative(state)

        self.store.save_checkpoint(state)
        result = self._result(state)
        self.store.save_result(result)
        return result

    def _result(self, state: PipelineState) -> PipelineResult:
        return PipelineResult.from_state(
            state,
            kb_fingerprint=self.kb_fp,
            prompt_fingerprint=self.prompt_fp,
            config=self.config.semantic_dict(),
            num_units=count_unique_units(self.focused_kb),
        )

    def run_iteration(self, state: PipelineState) -> None:
        """One iterative attempt: propose, process, select, record. No stop check.

        A failed proposal request or a stage-wide outage raises; nothing is
        recorded for the attempt and the run resumes from the last checkpoint.
        """
        before = USAGE.snapshot()
        state.iteration += 1
        attempt = state.attempts + 1  # the attempt being made
        with usage_stage("proposal"):
            proposed = self.proposer.propose_one(self._build_iterative_context(state, attempt))
        if not proposed:
            self._record(state, CandidateStatus.REFUSED)
        else:
            outcome = self.processor.process(proposed[0], state.embeddings, index=attempt)
            self._apply_outcome(state, outcome, increment_attempts=True)
        state.add_usage(usage_delta(USAGE.snapshot(), before))

    # ---- a0 ----

    def _run_batch(self, state: PipelineState) -> None:
        """One batch request; atomic (no mid-run checkpoint)."""
        before = USAGE.snapshot()
        state.iteration += 1
        videos = self.knowledge_base.by_video()
        budget = self.config.proposal_budget
        with usage_stage("proposal"):
            proposed = self.proposer.propose_batch(
                BatchContext(task=self.task, videos=videos, num_facets=budget,
                             seed=stage_seed(self.config.seed, "proposal"))
            )
        if not proposed:
            self._record(state, CandidateStatus.REFUSED)
            state.add_usage(usage_delta(USAGE.snapshot(), before))
            state.stop_reason = StopReason.BATCH_REFUSED
            return

        state.attempts += 1  # one batch request = one attempt
        kept, surplus = proposed[:budget], proposed[budget:]
        if surplus:
            logger.info("Batch returned %d facets; truncating to budget %d.",
                        len(proposed), budget)

        # Batched lifecycle, then selection in LLM output order.
        outcomes = self.processor.process_batch(kept, state.embeddings)
        for outcome in outcomes:
            self._apply_outcome(state, outcome, increment_attempts=False)
        for facet in surplus:
            self._record(state, CandidateStatus.TRUNCATED, facet=facet)

        state.add_usage(usage_delta(USAGE.snapshot(), before))
        state.stop_reason = StopReason.BATCH_COMPLETE

    # ---- a1 / a2 ----

    def _history(self, state: PipelineState) -> List[HistoryEntry]:
        pool_ids = set(state.pool_ids())
        titles = {r.facet_id: r.title for r in state.ledger if r.facet_id}
        history = []
        for r in state.ledger:
            if r.status in _NO_HISTORY_STATUSES or r.facet_id in pool_ids:
                continue
            outcome = ("replaced_out"
                       if r.status in (CandidateStatus.ACCEPTED, CandidateStatus.REPLACED)
                       else r.status.value)
            history.append(HistoryEntry(
                title=r.title or "", scope=r.scope or "", definition=r.definition or "",
                outcome=outcome,
                duplicate_of=titles.get(r.duplicate_of) if r.duplicate_of else None,
            ))
        return history

    def _items_videos(self, attempt: int):
        """Tutorial segments for the ITEMS component: all videos, or a sample
        of ``items_sample_videos`` drawn per attempt from the run seed."""
        videos = self.knowledge_base.by_video()
        n = self.config.items_sample_videos
        if n <= 0 or n >= len(videos):
            return videos
        rng = random.Random(stage_seed(self.config.seed, "items", attempt))
        chosen = set(rng.sample(sorted(videos), n))
        return {vid: items for vid, items in videos.items() if vid in chosen}

    def _build_iterative_context(self, state: PipelineState, attempt: int) -> IterativeContext:
        components = self.config.context
        return IterativeContext(
            task=self.task,
            components=components,
            pool=list(state.pool),
            annotations=state.annotations,
            knowledge_base=self.focused_kb,
            videos=(self._items_videos(attempt)
                    if ContextComponent.ITEMS in components else None),
            accepted_schemas=list(state.accepted_schemas),
            rejected_schemas=list(state.rejected_schemas),
            history=(self._history(state)
                     if ContextComponent.HISTORY in components else []),
            seed=stage_seed(self.config.seed, "proposal", attempt),
        )

    def _run_iterative(self, state: PipelineState) -> None:
        if state.pool and state.tree is None:
            self.evaluator.initialize(state)
        while (stop := self._stop_reason(state)) is None:
            self.run_iteration(state)
            self.store.save_checkpoint(state)
        state.stop_reason = stop
