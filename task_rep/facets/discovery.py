"""
Facet discovery and revision.

Uses batch-first VLM processing to propose new facets or revise existing ones.
Follows the same architectural pattern as VLMKnowledgeExtractor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from utils import (
    ModelConfig,
    batch_run_llm_calls,
    random_uid,
)

from knowledge_items import (
    KnowledgeItem,
)

from .prompts import (
    form_facet_discovery_request,
    process_facet_discovery_response,
    form_facet_revision_request,
    process_facet_revision_response,
)

from .schema import Facet


@dataclass
class VLMFacetExtractor:
    """
    Batch-first VLM-based facet extractor.

    Operations:
    - propose_facets: Discover new facets (single request)
    - revise_facets: Revise existing facets (single request)
    """
    model_config: ModelConfig
    max_concurrency: int = 32

    def propose_facets(
        self,
        task: str,
        videos: Dict[str, List[KnowledgeItem]],
        target_item_ids: List[str],
        facets: Optional[List[Facet]] = None,
        target_signature: Optional[str] = None,
    ) -> List[Facet]:
        """
        Propose new orthogonal facets to discriminate target items.

        Args:
            videos: Dict of video_id -> List of KnowledgeItems
            target_item_ids: List of target item IDs
            existing_facets: Previously discovered facets
            target_signature: Current shared signature of target items

        Returns:
            List of newly proposed facets
        """
        if facets is None:
            facets = []

        if target_signature is None:
            target_signature = ""

        batch_id = f"facet_discovery_{random_uid()}"

        target_items = [
            item
            for video_items in videos.values()
            for item in video_items
            if item.id in target_item_ids
        ]

        request_args = [{
            "task": task,
            "videos": videos,
            "target_items": target_items,
            "facets": facets,
            "target_signature": target_signature,
        }]

        batch_results = batch_run_llm_calls(
            batch_id,
            request_args,
            form_facet_discovery_request,
            process_facet_discovery_response,
            max_concurrency=self.max_concurrency,
            model_config=self.model_config,
        )

        return [facet for result in batch_results for facet in result]

    def revise_facets(
        self,
        task: str,
        facets: List[Facet],
    ) -> List[Facet]:
        """
        Revise existing facets 

        Operates corpus-level: single request with all annotated items.

        This can:
        - Refine definitions and guidelines to adjust the granularity
        - Split overly broad facets
        - Merge redundant facets

        Args:
            facets: Current facets to revise

        Returns:
            List of revised facets
        """
        if not facets:
            return []

        batch_id = f"facet_revision_{random_uid()}"

        request_args = [{"task": task, "facets": facets}]

        batch_results = batch_run_llm_calls(
            batch_id,
            request_args,
            form_facet_revision_request,
            process_facet_revision_response,
            max_concurrency=self.max_concurrency,
            model_config=self.model_config,
        )

        return [facet for result in batch_results for facet in result]