"""
Facet vocabulary building.

Uses batch-first VLM processing to propose and consolidate vocabulary labels
for facets. Two-phase process:
1. Per-(facet, video) proposal: parallel via batch_run_llm_calls
2. Per-facet consolidation: parallel via batch_run_llm_calls

Follows the same architectural pattern as VLMKnowledgeExtractor and VLMFacetExtractor.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

from task_rep.knowledge_items.schema import KnowledgeBase

from task_rep.utils.llm import random_uid
from task_rep.utils.llm import batch_run_llm_calls
from task_rep.utils.llm import MessageContent

from task_rep.prompts.fa_facet_vocabulary import (
    form_vocabulary_proposal_request,
    process_vocabulary_proposal_response,
    form_vocabulary_consolidation_request,
    process_vocabulary_consolidation_response,
)

from .schema import Facet, FacetLabel


@dataclass
class VLMFacetVocabularyBuilder:
    """
    Batch-first VLM-based facet vocabulary builder.

    Two-phase process:
    1. Per-(facet, video) proposal: For each facet and video, propose vocabulary
       labels based on the knowledge items in that video. All (facet, video) pairs are processed in parallel.
    2. Per-facet consolidation: Merge per-facet proposals into canonical vocabulary. All facets are processed in parallel.

    build_vocabulary(facets, knowledge_base) -> List[Facet]
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

    def build_vocabulary(
        self,
        facets: List[Facet],
        knowledge_base: KnowledgeBase,
    ) -> List[Facet]:
        """
        Build vocabulary for each facet using knowledge items from the corpus.

        Args:
            facets: Facets with definitions/guidelines (vocabulary may be empty or partial).
            knowledge_base: Knowledge items to derive vocabulary from.

        Returns:
            New list of Facets with updated vocabulary fields.
        """
        if not facets:
            return []

        videos = knowledge_base.by_video()
        if not videos:
            return facets

        # Pre-format video items as MessageContent (reused across all facets)
        video_contents: Dict[str, List[MessageContent]] = {}
        for video_id, video_items in videos.items():
            video_contents[video_id] = [item.format_to_message_content() for item in video_items]

        # Phase 1: Propose vocabulary labels — one request per (facet, video) pair
        batch_id = f"vocab_proposal_{random_uid()}"
        request_keys = []  # track (facet_id, video_id) for each request
        request_args = []

        for facet in facets:
            facet_mc = facet.format_to_message_content()
            for video_id, video_content in video_contents.items():
                request_keys.append((facet.id, video_id))
                request_args.append({
                    "task": self.task,
                    "facet": facet_mc,
                    "video_items": video_content,
                })

        batch_results = batch_run_llm_calls(
            batch_id,
            request_args,
            form_vocabulary_proposal_request,
            process_vocabulary_proposal_response,
            max_concurrency=self.max_concurrency,
            provider=self.provider,
            model=self.model,
        )

        # Re-group proposal results by facet
        proposals_by_facet: Dict[str, Dict[str, List[FacetLabel]]] = defaultdict(dict)
        for (facet_id, video_id), result in zip(request_keys, batch_results):
            proposals_by_facet[facet_id][video_id] = [
                FacetLabel(label=lbl["label"], definition=lbl["definition"])
                for lbl in result
            ]

        # Phase 2: Per-facet consolidation
        batch_id = f"vocab_consolidation_{random_uid()}"

        request_args = []

        for facet in facets:
            facet_proposals = proposals_by_facet.get(facet.id, {})
            proposals_by_video_content = []
            for video_id, proposals in facet_proposals.items():
                proposals_by_video_content.append({
                    "type": "text",
                    "text": f"Video {video_id}:",
                })
                for proposal in proposals:
                    proposals_by_video_content.append({
                        "type": "text",
                        "text": str(proposal),
                    })

            request_args.append({
                "task": self.task,
                "facet": facet.format_to_message_content(),
                "existing_vocabulary": facet.format_vocabulary(),
                "proposals_by_video": MessageContent(content=proposals_by_video_content),
            })

        batch_results = batch_run_llm_calls(
            batch_id,
            request_args,
            form_vocabulary_consolidation_request,
            process_vocabulary_consolidation_response,
            max_concurrency=self.max_concurrency,
            provider=self.provider,
            model=self.model,
        )

        updated_facets = []
        for facet, result in zip(facets, batch_results):
            updated_vocabulary = [
                FacetLabel(label=lbl["label"], definition=lbl["definition"])
                for lbl in result
            ]
            facet.replace_vocabulary(updated_vocabulary)
            updated_facets.append(facet)

        return updated_facets
