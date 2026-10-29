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
                         None = use module default (0.0).

        Returns:
            AnnotationTable mapping (item_id, facet_id) -> label.
        """
        if not facets:
            return AnnotationTable()

        videos = knowledge_base.by_video()
        if not videos:
            return AnnotationTable()

        # One request per (facet, video) pair, flattened into a single batch
        batch_id = f"facet_labeling_{random_uid()}"
        request_keys: List[Tuple[str, str]] = []  # (facet_id, video_id)
        request_args = []

        for facet in facets:
            for video_id, video_items in videos.items():
                request_keys.append((facet.id, video_id))
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
        )

        # Collect results into AnnotationTable
        annotations = AnnotationTable()
        for (facet_id, _video_id), result in zip(request_keys, batch_results):
            for entry in result:
                item_id = entry["item_id"]
                label = entry["label"]
                if item_id and label:
                    annotations[(item_id, facet_id)] = label

        return annotations
