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
from typing import Any, Dict, List, Optional

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

from .schema import Facet

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
    # id of the latest request's log in static/lm-batches (for transcripts)
    last_batch_id: Optional[str] = None

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
        batch_id = self.last_batch_id = f"facet_init_{random_uid()}"

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

    def propose_facet_iterative(self, seed: Optional[int] = None, **request: Any) -> List[Facet]:
        """Propose exactly one new facet conditioned on the existing pool.

        ``request`` holds the arguments of ``form_facet_iterative_request``
        (task, pool, context components, corpus, and the components' data).
        Returns a list with the new Facet, or an empty list if the LLM refused.
        """
        batch_id = self.last_batch_id = f"facet_iterative_{random_uid()}"
        request_args = [request]

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
