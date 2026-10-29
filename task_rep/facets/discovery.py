"""
Facet proposal.

Uses batch-first VLM processing to propose facets in one batch (a0) or one at
a time conditioned on the current pool (a1/a2). Follows the same architectural
pattern as VLMKnowledgeExtractor.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any, Collection, Dict, List, Optional

from utils import (
    ModelConfig,
    batch_run_llm_calls,
    random_uid,
)

from knowledge_items import (
    KnowledgeItem,
)

from .prompts import (
    form_facet_init_request,
    process_facet_init_response,
    form_facet_iterative_request,
    process_facet_iterative_response,
)

from .schema import Facet, AnnotationTable

logger = logging.getLogger(__name__)

@dataclass
class VLMFacetExtractor:
    """
    Batch-first VLM-based facet extractor.

    Operations:
    - init_facets: Propose a batch of facets from tutorials (single request)
    - propose_facet_iterative: Propose one facet given the current pool (single request)
    """
    model_config: ModelConfig
    max_concurrency: int = 32

    def init_facets(
        self,
        task: str,
        videos: Dict[str, List[KnowledgeItem]],
        num_facets: int,
        seed: Optional[int] = None,
    ) -> List[Facet]:
        """
        Bootstrap the algorithm with common-sense contextual factors.

        Args:
            task: Task name/description
            videos: Dict of video_id -> List of KnowledgeItems
            num_facets: Number of facets to initialize with
            seed: Sampling seed for the request (None = provider default)

        Returns:
            List of initialized facets
        """
        batch_id = f"facet_init_{random_uid()}"

        # Sample a (all) tutorials with a few items each for concrete examples
        sample_n_videos = len(videos)
        video_ids = list(videos.keys())
        if len(video_ids) > sample_n_videos:
            video_ids = random.sample(video_ids, sample_n_videos)
        sampled_videos = {}
        for vid in video_ids:
            sampled_videos[vid] = videos[vid]

        request_args = [{
            "task": task,
            "videos": sampled_videos,
            "num_facets": num_facets,
        }]

        batch_results = batch_run_llm_calls(
            batch_id,
            request_args,
            form_facet_init_request,
            process_facet_init_response,
            max_concurrency=self.max_concurrency,
            model_config=self.model_config,
            seeds=[seed],
        )

        return [facet for result in batch_results for facet in result]

    def propose_facet_iterative(
        self,
        task: str,
        facets: List[Facet],
        components: Collection[str],
        videos: Optional[Dict[str, List[KnowledgeItem]]] = None,
        annotations: Optional[AnnotationTable] = None,
        knowledge_base: Optional[Any] = None,
        accepted_schemas: Optional[List[Any]] = None,
        rejected_schemas: Optional[List[Any]] = None,
        history: Optional[List[Any]] = None,
        seed: Optional[int] = None,
    ) -> List[Facet]:
        """Propose exactly one new facet conditioned on the existing pool.

        Args:
            task: Task name/description
            facets: Current facet pool (definitions + vocabularies shown to LLM)
            components: Context component names to include ("facets", "items", "schemas", "history")
            videos: Video segments with their pool annotations (included with "items")
            annotations: Current pool annotations for the video segments
            knowledge_base: Active optimization scope for frequency statistics
            accepted_schemas: Surviving schema summaries (included with "schemas")
            rejected_schemas: Rejected schema summaries (included with "schemas")
            history: Previously tried, non-kept candidates (included with "history")
            seed: Sampling seed for the request (None = provider default)

        Returns:
            A list of new Facets, or an empty list if the LLM refused.
        """
        batch_id = f"facet_iterative_{random_uid()}"

        request_args = [{
            "task": task,
            "facets": facets,
            "components": components,
            "videos": videos,
            "annotations": annotations,
            "knowledge_base": knowledge_base,
            "accepted_schemas": accepted_schemas,
            "rejected_schemas": rejected_schemas,
            "history": history,
        }]

        batch_results = batch_run_llm_calls(
            batch_id,
            request_args,
            form_facet_iterative_request,
            process_facet_iterative_response,
            max_concurrency=self.max_concurrency,
            model_config=self.model_config,
            seeds=[seed],
        )

        return [facet for result in batch_results for facet in result]
