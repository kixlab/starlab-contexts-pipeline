"""
Facet labeling.

Uses batch-first VLM processing to assign vocabulary labels to knowledge items.
One request per (facet, video) pair, all processed in a single batch.

Follows the same architectural pattern as VLMFacetVocabularyBuilder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from knowledge_items.schema import KnowledgeBase

from utils import (
    ModelConfig,
    batch_run_llm_calls,
    random_uid,
    MessageContent,
)

from prompts import (
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
    ) -> AnnotationTable:
        """
        Label knowledge items with facet vocabulary labels.

        Args:
            facets: Facets with vocabulary to label against.
            knowledge_base: Knowledge items to label.

        Returns:
            AnnotationTable mapping (item_id, facet_id) -> label.
        """
        if not facets:
            return AnnotationTable()

        videos = knowledge_base.by_video()
        if not videos:
            return AnnotationTable()

        # Pre-format video items as MessageContent (reused across all facets)
        video_contents: Dict[str, List[MessageContent]] = {}
        for video_id, video_items in videos.items():
            video_contents[video_id] = [
                item.format_to_message_content() for item in video_items
            ]

        # One request per (facet, video) pair, flattened into a single batch
        batch_id = f"facet_labeling_{random_uid()}"
        request_keys: List[Tuple[str, str]] = []  # (facet_id, video_id)
        request_args = []

        for facet in facets:
            facet_mc = facet.format_to_message_content()
            for video_id, video_content in video_contents.items():
                request_keys.append((facet.id, video_id))
                request_args.append({
                    "task": task,
                    "facet": facet_mc,
                    "video_items": video_content,
                })

        batch_results = batch_run_llm_calls(
            batch_id,
            request_args,
            form_labeling_request,
            process_labeling_response,
            max_concurrency=self.max_concurrency,
            model_config=self.model_config,
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
