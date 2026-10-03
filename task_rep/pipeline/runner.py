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

from utils.llm import USAGE, usage_delta, usage_stage

from optimization import count_unique_units

from contextual import bookkeeping
from contextual.evaluation import PoolEvaluator
from contextual.lifecycle import CandidateProcessor, Embedder, VocabularyBuilder, Labeler
from contextual.models import RunState, RunResult, CandidateStatus, StopReason
from contextual.settings import stage_seed
from contextual.store import RunStore, kb_fingerprint

from .config import PipelineConfig, ProposalStrategy, ContextComponent
from .fingerprint import prompt_fingerprint
from .proposal import FacetProposer, BatchContext, IterativeContext, HistoryEntry

logger = logging.getLogger(__name__)

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
        self.store = RunStore(
            cache_dir=cache_dir, config=config,
            kb_fp=self.kb_fp, prompt_fp=self.prompt_fp, prefix="pipeline",
        )

    # ---- public entry points ----

    def run(self) -> RunResult:
        state: Optional[RunState] = None
        if self.config.use_cache:
            state = self.store.restore_checkpoint()
        if state is None:
            state = RunState(fingerprint=self.store.fingerprint)

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

    def _result(self, state: RunState) -> RunResult:
        return RunResult.from_state(
            state,
            kb_fingerprint=self.kb_fp,
            prompt_fingerprint=self.prompt_fp,
            config=self.config.semantic_dict(),
            num_units=count_unique_units(self.focused_kb),
        )

    def run_iteration(self, state: RunState) -> None:
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
            bookkeeping.record(state, CandidateStatus.REFUSED)
        else:
            outcome = self.processor.process(proposed[0], state.embeddings, index=attempt)
            bookkeeping.apply_outcome(state, self.evaluator, outcome, increment_attempts=True)
        state.add_usage(usage_delta(USAGE.snapshot(), before))

    # ---- a0 ----

    def _run_batch(self, state: RunState) -> None:
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
            bookkeeping.record(state, CandidateStatus.REFUSED)
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
            bookkeeping.apply_outcome(state, self.evaluator, outcome, increment_attempts=False)
        for facet in surplus:
            bookkeeping.record(state, CandidateStatus.TRUNCATED, facet=facet)

        state.add_usage(usage_delta(USAGE.snapshot(), before))
        state.stop_reason = StopReason.BATCH_COMPLETE

    # ---- a1 / a2 ----

    def _history(self, state: RunState) -> List[HistoryEntry]:
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

    def _build_iterative_context(self, state: RunState, attempt: int) -> IterativeContext:
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

    def _run_iterative(self, state: RunState) -> None:
        if state.pool and state.tree is None:
            self.evaluator.initialize(state)
        while (stop := bookkeeping.stop_reason(state, self.config)) is None:
            self.run_iteration(state)
            self.store.save_checkpoint(state)
        state.stop_reason = stop
