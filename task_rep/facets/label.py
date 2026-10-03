"""
Facet labeling.

Uses batch-first VLM processing to assign vocabulary labels to knowledge items.
One request per (facet, video) pair, all processed in a single batch.

Follows the same architectural pattern as VLMFacetVocabularyBuilder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from knowledge_items.schema import KnowledgeBase

from utils import (
    ModelConfig,
    batch_run_llm_calls,
    random_uid,
)

from .prompts import (
    form_labeling_request,
    process_labeling_response,
)
from .schema import (
    AnnotationTable,
    Facet,
)

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
    ) -> AnnotationTable:
        """
        Label knowledge items with facet vocabulary labels.

        Args:
            facets: Facets with vocabulary to label against.
            knowledge_base: Knowledge items to label.
            temperature: Optional temperature override for LLM calls.
                         None = provider default.

        Returns:
            AnnotationTable mapping (item_id, facet_id) -> label.
        """
        return self.label_batch_multi(
            task, facets, knowledge_base, seeds=[None], temperature=temperature,
        )[0]

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
        (seed, facet, video) request runs concurrently.

        Args:
            seeds: One sampling seed per pass (None = provider default).
            temperature: Optional temperature override (None = provider default).

        Returns:
            One AnnotationTable per seed, in ``seeds`` order.
        """
        if not facets:
            return [AnnotationTable() for _ in seeds]

        videos = knowledge_base.by_video()
        if not videos:
            return [AnnotationTable() for _ in seeds]

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
        )

        # Collect results into one AnnotationTable per pass
        passes = [AnnotationTable() for _ in seeds]
        for (pass_idx, facet_id), result in zip(request_keys, batch_results):
            for entry in result:
                item_id = entry["item_id"]
                label = entry["label"]
                if item_id and label:
                    passes[pass_idx][(item_id, facet_id)] = label

        return passes
