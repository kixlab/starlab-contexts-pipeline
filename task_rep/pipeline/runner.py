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
from typing import Dict, Optional

from knowledge_items import KnowledgeBase

from utils.llm import USAGE, usage_delta, usage_stage

from optimization import count_unique_units

from contextual import bookkeeping
from contextual.corpus import evaluated_segments, select_corpus
from contextual.evaluation import PoolEvaluator
from contextual.feedback import factor_ids, history_entries
from contextual.lifecycle import CandidateProcessor, Embedder, VocabularyBuilder, Labeler
from contextual.models import RunState, RunResult, CandidateStatus, StopReason, refinement_edges
from contextual.settings import stage_seed
from contextual import cache

from .config import PipelineConfig, ProposalStrategy, ContextComponent
from .fingerprint import prompt_fingerprint
from .proposal import FacetProposer, BatchContext, IterativeContext

logger = logging.getLogger(__name__)

KIND = "pipeline"


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
        runs_dir: str | Path,
        name: str = "run",
        sources: Optional[Dict[str, str]] = None,
        fresh: bool = False,
    ):
        self.config = config
        self.task = task
        knowledge_base = select_corpus(knowledge_base, config)
        self.knowledge_base = knowledge_base

        # Evaluated segments: every prompt block and the design space use them;
        # vocabulary and labeling see the whole selected corpus.
        self.focused_kb = evaluated_segments(knowledge_base, config)
        self.corpus = self.focused_kb.by_video()

        self.proposer = proposer
        self.processor = CandidateProcessor(
            config=config, task=task, knowledge_base=knowledge_base,
            vocab_builder=vocab_builder, labeler=labeler, embedder=embedder,
        )
        self.evaluator = PoolEvaluator(config=config, focused_kb=self.focused_kb)

        self.kb_fp = cache.kb_fingerprint(knowledge_base)
        self.prompt_fp = prompt_fingerprint(config)
        self.fingerprint = cache.run_fingerprint(config, self.kb_fp, self.prompt_fp)
        # the run folder (contextual.cache): resumed if interrupted, else new
        self.runs_dir, self.name, self.sources = Path(runs_dir), name, sources or {}
        self.fresh = fresh or not config.use_cache
        self.folder: Optional[cache.RunFolder] = None

    # ---- public entry points ----

    def run(self) -> RunResult:
        self.folder = cache.RunFolder.open(
            self.runs_dir, self.name, KIND, self.fingerprint, fresh=self.fresh,
            info={"config": self.config.semantic_dict(), "kb_fingerprint": self.kb_fp,
                  "prompt_fingerprint": self.prompt_fp, "sources": self.sources})
        with self.folder.logging():
            state = cache.start(self.folder, KIND, self.config, self.fingerprint)
            if state.pool:
                self.evaluator.initialize(state)
            try:
                if state.stop_reason is None:
                    if self.config.strategy == ProposalStrategy.BATCH:
                        self._step(state, self._run_batch)
                    else:
                        self._run_iterative(state)
            except KeyboardInterrupt:
                logger.warning("Interrupted during attempt %d; running %s again resumes "
                               "after attempt %d.", state.attempts + 1, self.name, state.attempts)
                raise
            result = self._result(state)
            self.folder.finish(result)
            cache.log_end(self.folder, result)
        return result

    def _step(self, state: RunState, work) -> None:
        cache.run_step(self.folder, state, work, lambda s: (
            f"attempt {s.attempts} (proposals {s.proposals}/{self.config.proposal_budget})"))

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
        recorded for the attempt and rerunning resumes from the last completed one.
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
        """One batch request; one checkpoint step (an interruption redoes it)."""
        before = USAGE.snapshot()
        state.iteration += 1
        budget = self.config.proposal_budget
        with usage_stage("proposal"):
            proposed = self.proposer.propose_batch(
                BatchContext(task=self.task, videos=self.corpus, num_facets=budget,
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

    def _items_video_ids(self, attempt: int):
        """Videos whose labels ITEMS shows: all (None), or a sample of
        ``items_sample_videos`` drawn per attempt from the run seed."""
        n = self.config.items_sample_videos
        if n <= 0 or n >= len(self.corpus):
            return None
        rng = random.Random(stage_seed(self.config.seed, "items", attempt))
        return sorted(rng.sample(sorted(self.corpus), n))

    def _build_iterative_context(self, state: RunState, attempt: int) -> IterativeContext:
        components = self.config.context
        return IterativeContext(
            task=self.task,
            components=components,
            corpus=self.corpus,
            pool=list(state.pool),
            ids=factor_ids(state),
            annotations=state.annotations,
            knowledge_base=self.focused_kb,
            item_video_ids=(self._items_video_ids(attempt)
                            if ContextComponent.ITEMS in components else None),
            accepted_schemas=list(state.accepted_schemas),
            rejected_schemas=list(state.rejected_schemas),
            refinement_edges=refinement_edges(state),
            refinement_threshold=self.config.refinement_threshold,
            max_schema_facets=self.config.max_schema_facets,
            history=(history_entries(state)
                     if ContextComponent.HISTORY in components else []),
            seed=stage_seed(self.config.seed, "proposal", attempt),
        )

    def _run_iterative(self, state: RunState) -> None:
        while (stop := bookkeeping.stop_reason(state, self.config)) is None:
            self._step(state, self.run_iteration)
        state.stop_reason = stop
