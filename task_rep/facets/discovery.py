"""
Facet discovery and revision.

Uses batch-first VLM processing to propose new facets or revise existing ones.
Follows the same architectural pattern as VLMKnowledgeExtractor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from knowledge_items.schema import KnowledgeItem

from utils import (
    ModelConfig,
    batch_run_llm_calls,
    random_uid,
    MessageContent,
)

from prompts import (
    form_facet_discovery_request,
    process_facet_discovery_response,
    form_facet_revision_request,
    process_facet_revision_response,
)

from .schema import (
    Facet,
    FacetLabel,
)


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

        # Build request args
        batch_id = f"facet_discovery_{random_uid()}"
        request_args = []

        video_contents = {}
        facets_contents = {}
        target_items_contents = {}

        for video_id, video_items in videos.items():
            video_content = []
            for item in video_items:
                item_content = item.format_to_message_content()
                if item.id in target_item_ids:
                    target_items_contents[item.id] = item_content
                video_content.extend(item_content.to_content())

            video_contents[video_id] = MessageContent(content=video_content)

        for facet in facets:
            facet_content = facet.format_to_message_content()
            facets_contents[facet.id] = facet_content

        request_args = [{
            "task": task,
            "tutorial_contents": video_contents,
            "target_items": target_items_contents,
            "facets": facets_contents,
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

        new_facets = []
        for result in batch_results:
            for idx in range(len(result)):
                candidate = result[idx]
                new_vocabulary = [FacetLabel(
                    label=ld["label"],
                    definition=ld["definition"],
                ) for ld in candidate.get("segment_labels", [])]
                new_facet = Facet(
                    id=random_uid(),
                    title=candidate["title"],
                    title_plural=candidate["title_plural"],
                    scope=candidate["scope"],
                    definition=candidate["definition"],
                    justification=candidate["justification"],
                    guidelines=candidate["guidelines"],
                    vocabulary=new_vocabulary,
                )
                new_facets.append(new_facet)

        return new_facets

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

        # Format existing facets
        facets_contents = {}
        for facet in facets:
            facets_contents[facet.id] = facet.format_to_message_content()

        request_args = [{
            "task": task,
            "facets": facets_contents,
        }]

        batch_results = batch_run_llm_calls(
            batch_id,
            request_args,
            form_facet_revision_request,
            process_facet_revision_response,
            max_concurrency=self.max_concurrency,
            model_config=self.model_config,
        )

        new_facets = []
        for result in batch_results:
            for idx in range(len(result)):
                candidate = result[idx]
                new_vocabulary = [FacetLabel(
                    label=ld["label"],
                    definition=ld["definition"],
                ) for ld in candidate.get("segment_labels", [])]
                new_facet = Facet(
                    id=random_uid(),
                    title=candidate["title"],
                    title_plural=candidate["title_plural"],
                    scope=candidate["scope"],
                    definition=candidate["definition"],
                    justification=candidate["justification"],
                    guidelines=candidate["guidelines"],
                    vocabulary=new_vocabulary,
                )
                new_facets.append(new_facet)
        return new_facets