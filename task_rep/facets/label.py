"""
Facet labeling.

Uses batch-first VLM processing to assign vocabulary labels to knowledge items.
One request per (facet, video) pair, all processed in a single batch.

Follows the same architectural pattern as VLMFacetVocabularyBuilder.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

from knowledge_items.schema import KnowledgeBase

from utils import (
    ModelConfig,
    batch_run_llm_calls,
    random_uid,
)
from utils.llm import LLMRequestFailed

from .prompts import (
    form_labeling_request,
    process_labeling_response,
)
from .schema import (
    AnnotationTable,
    Facet,
)

logger = logging.getLogger(__name__)

@dataclass
class VLMFacetLabeler:
    """
    Batch-first VLM-based facet labeler.

    One request per (facet, video) pair. All pairs are flattened into
    a single batch_run_llm_calls invocation for maximum parallelism.

    label_batch(facets, knowledge_base) -> AnnotationTable
    label_batch_multi(facets, knowledge_base, seeds) -> List[AnnotationTable]
    """
    model_config: ModelConfig
    max_concurrency: int = 32

    def label_batch(
        self,
        task: str,
        facets: List[Facet],
        knowledge_base: KnowledgeBase,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> AnnotationTable:
        """
        Label knowledge items with facet vocabulary labels.

        A facet with any permanently failed request is left out of the result
        entirely (never partially labeled).

        Args:
            facets: Facets with vocabulary to label against.
            knowledge_base: Knowledge items to label.
            temperature: Optional temperature override for LLM calls.
                         None = provider default.
            seed: Sampling seed (None = provider default).

        Returns:
            AnnotationTable mapping (item_id, facet_id) -> label.
        """
        passes, failed = self._label(task, facets, knowledge_base, [seed], temperature)
        failed_facets = {facet_id for _, facet_id in failed}
        if failed_facets:
            logger.warning("Labeling failed for facet(s) %s.", sorted(failed_facets))
        return passes[0].filter_facets(
            [f.id for f in facets if f.id not in failed_facets]
        )

    def label_batch_multi(
        self,
        task: str,
        facets: List[Facet],
        knowledge_base: KnowledgeBase,
        seeds: List[Optional[int]],
        temperature: Optional[float] = None,
    ) -> List[AnnotationTable]:
        """
        Label knowledge items once per seed, all in a single batch.

        Used for independent labeling passes (e.g. IRR raters): every
        (seed, facet, video) request runs concurrently. Items of permanently
        failed requests are simply missing from that pass.

        Args:
            seeds: One sampling seed per pass (None = provider default).
            temperature: Optional temperature override (None = provider default).

        Returns:
            One AnnotationTable per seed, in ``seeds`` order.
        """
        passes, _ = self._label(task, facets, knowledge_base, seeds, temperature)
        return passes

    def _label(
        self,
        task: str,
        facets: List[Facet],
        knowledge_base: KnowledgeBase,
        seeds: List[Optional[int]],
        temperature: Optional[float],
    ) -> Tuple[List[AnnotationTable], Set[Tuple[int, str]]]:
        """Run all (pass, facet, video) requests; return passes and failed (pass, facet)."""
        passes = [AnnotationTable() for _ in seeds]
        failed: Set[Tuple[int, str]] = set()
        if not facets:
            return passes, failed

        videos = knowledge_base.by_video()
        if not videos:
            return passes, failed

        # One request per (pass, facet, video), flattened into a single batch
        batch_id = f"facet_labeling_{random_uid()}"
        request_keys: List[Tuple[int, str]] = []  # (pass index, facet_id)
        request_args = []
        request_seeds: List[Optional[int]] = []

        for pass_idx, seed in enumerate(seeds):
            for facet in facets:
                for video_items in videos.values():
                    request_keys.append((pass_idx, facet.id))
                    request_seeds.append(seed)
                    request_args.append({
                        "task": task,
                        "facet": facet,
                        "video_items": video_items,
                    })

        batch_results = batch_run_llm_calls(
            batch_id,
            request_args,
            form_labeling_request,
            process_labeling_response,
            max_concurrency=self.max_concurrency,
            model_config=self.model_config,
            temperature=temperature,
            seeds=request_seeds,
            allow_errors=True,
        )

        for (pass_idx, facet_id), result in zip(request_keys, batch_results):
            if isinstance(result, LLMRequestFailed):
                failed.add((pass_idx, facet_id))
                continue
            for entry in result:
                item_id = entry["item_id"]
                label = entry["label"]
                if item_id and label:
                    passes[pass_idx][(item_id, facet_id)] = label

        return passes, failed
