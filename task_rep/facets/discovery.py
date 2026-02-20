"""
Facet discovery and revision.

Uses batch-first VLM processing to propose new facets or revise existing ones.
Follows the same architectural pattern as VLMKnowledgeExtractor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from task_rep.knowledge_items.schema import KnowledgeItem

from task_rep.utils.llm import random_uid
from task_rep.utils.llm import batch_run_llm_calls
from task_rep.utils.llm import MessageContent

from task_rep.prompts.fa_facet_discovery import (
    form_facet_discovery_request,
    process_facet_discovery_response,
)
from task_rep.prompts.fa_facet_revision import (
    form_facet_revision_request,
    process_facet_revision_response,
)

from .schema import Facet, FacetLabel


@dataclass
class VLMFacetExtractor:
    """
    Batch-first VLM-based facet extractor.

    Operations:
    - propose_facets: Discover new facets (single request)
    - revise_facets: Revise existing facets (single request)
    """

    task: str = ""
    provider: Optional[str] = None
    model: Optional[str] = None
    max_concurrency: int = 32

    def __post_init__(self):
        """Initialize default model based on provider."""
        if self.provider is None:
            self.provider = "openai"
        if self.model is None:
            if self.provider == "openai":
                self.model = "gpt-4o-mini"
            elif self.provider == "anthropic":
                self.model = "us.anthropic.claude-3-5-haiku-20241022-v1:0"

    def propose_facets(
        self,
        videos: Dict[str, List[KnowledgeItem]],
        target_item_ids: List[str],
        existing_facets: Optional[List[Facet]] = None,
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
        if existing_facets is None:
            existing_facets = []

        if target_signature is None:
            target_signature = ""

        # Build request args
        batch_id = f"facet_discovery_{random_uid()}"
        request_args = []

        video_contents = {}
        existing_facets_contents = {}
        target_items_contents = {}

        for video_id, video_items in videos.items():
            video_content = []
            for item in video_items:
                item_content = item.format_to_message_content()
                if item.id in target_item_ids:
                    target_items_contents[item.id] = item_content
                video_content.append(item_content.to_content())

            video_contents[video_id] = MessageContent(content=video_content)

        for facet in existing_facets:
            facet_content = facet.format_to_message_content()
            existing_facets_contents[facet.id] = facet_content

        request_args = [{
            "task": self.task,
            "tutorial_contents": video_contents,
            "target_items": target_items_contents,
            "existing_facets": existing_facets_contents,
            "target_signature": target_signature,
        }]

        batch_results = batch_run_llm_calls(
            batch_id,
            request_args,
            form_facet_discovery_request,
            process_facet_discovery_response,
            max_concurrency=self.max_concurrency,
            provider=self.provider,
            model=self.model,
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
        existing_facets: List[Facet],
    ) -> List[Facet]:
        """
        Revise existing facets 

        Operates corpus-level: single request with all annotated items.

        This can:
        - Refine definitions and guidelines to adjust the granularity
        - Split overly broad facets
        - Merge redundant facets

        Args:
            existing_facets: Current facets to revise

        Returns:
            List of revised facets
        """
        if not existing_facets:
            return []

        batch_id = f"facet_revision_{random_uid()}"

        # Format existing facets
        existing_facets_contents = {}
        for facet in existing_facets:
            existing_facets_contents[facet.id] = facet.format_to_message_content()

        request_args = [{
            "task": self.task,
            "existing_facets": existing_facets_contents,
        }]

        batch_results = batch_run_llm_calls(
            batch_id,
            request_args,
            form_facet_revision_request,
            process_facet_revision_response,
            max_concurrency=self.max_concurrency,
            provider=self.provider,
            model=self.model,
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