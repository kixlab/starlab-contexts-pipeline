"""
Facet vocabulary building.

Uses batch-first VLM processing to propose and consolidate vocabulary labels
for facets. Two-phase process:
1. Per-(facet, video) proposal: parallel via batch_run_llm_calls
2. Per-facet consolidation: parallel via batch_run_llm_calls

Follows the same architectural pattern as VLMKnowledgeExtractor and VLMFacetExtractor.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

from knowledge_items.schema import KnowledgeBase

from utils import (
    ModelConfig,
    batch_run_llm_calls,
    random_uid,
)

from .prompts import (
    form_vocabulary_proposal_request,
    process_vocabulary_proposal_response,
    form_vocabulary_consolidation_request,
    process_vocabulary_consolidation_response,
)

from .schema import (
    Facet,
    FacetLabel,
)

logger = logging.getLogger(__name__)


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
    model_config: ModelConfig
    max_concurrency: int = 32

    def build_vocabulary(
        self,
        task: str,
        facets: List[Facet],
        knowledge_base: KnowledgeBase,
        sample_videos: int = 0,
    ) -> List[Facet]:
        """
        Build vocabulary for each facet using knowledge items from the corpus.

        Args:
            facets: Facets with definitions/guidelines (vocabulary may be empty or partial).
            knowledge_base: Knowledge items to derive vocabulary from.
            sample_videos: Number of videos to derive the vocabulary (`0` means all)

        Returns:
            New list of Facets with updated vocabulary fields.
        """
        if not facets:
            return []

        videos = knowledge_base.by_video()
        if not videos:
            return facets

        # Phase 1: Propose vocabulary labels — one request per (facet, video) pair
        batch_id = f"vocab_proposal_{random_uid()}"
        request_keys = []  # track (facet_id, video_id) for each request
        request_args = []

        video_items_list = list(videos.items())
        if sample_videos > 0 and sample_videos < len(video_items_list):
            video_items_list = random.sample(video_items_list, sample_videos)
            logger.info(
                "Vocab proposal: sampling %d/%d videos.", sample_videos, len(videos)
            )

        for facet in facets:
            for video_id, video_items in video_items_list:
                request_keys.append((facet.id, video_id))
                request_args.append({
                    "task": task,
                    "facet": facet,
                    "video_items": video_items,
                })

        batch_results = batch_run_llm_calls(
            batch_id,
            request_args,
            form_vocabulary_proposal_request,
            process_vocabulary_proposal_response,
            max_concurrency=self.max_concurrency,
            model_config=self.model_config,
        )

        # Re-group proposal results by facet
        proposals_by_facet: Dict[str, Dict[str, List[FacetLabel]]] = defaultdict(dict)
        for (facet_id, video_id), result in zip(request_keys, batch_results):
            if result:  # only store non-empty proposals
                proposals_by_facet[facet_id][video_id] = result

        # Phase 2: Per-facet consolidation (revises definition, guidelines, and vocabulary together)
        batch_id = f"vocab_consolidation_{random_uid()}"

        request_args = []
        consolidated_facet_ids: List[str] = []  # facets that have a consolidation request, in request order

        for facet in facets:
            proposals_by_video = proposals_by_facet.get(facet.id, {})
            if not proposals_by_video:
                logger.info(
                    "No vocabulary proposals for facet %s; skipping consolidation.",
                    facet.id,
                )
                continue
            consolidated_facet_ids.append(facet.id)
            request_args.append({
                "task": task,
                "facet": facet,
                "proposals_by_video": proposals_by_video,
            })

        batch_results = batch_run_llm_calls(
            batch_id,
            request_args,
            form_vocabulary_consolidation_request,
            process_vocabulary_consolidation_response,
            max_concurrency=self.max_concurrency,
            model_config=self.model_config,
        )

        # process_vocabulary_consolidation_response returns a fully-revised Facet.
        revised_by_id: Dict[str, Facet] = dict(zip(consolidated_facet_ids, batch_results))
        return [revised_by_id.get(facet.id, facet) for facet in facets]
