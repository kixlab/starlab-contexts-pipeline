"""
Facet discovery and revision.

Uses batch-first VLM processing to propose new facets or revise existing ones.
Follows the same architectural pattern as VLMKnowledgeExtractor.
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
    form_facet_discovery_request,
    process_facet_discovery_response,
    form_facet_init_request,
    process_facet_init_response,
    form_facet_revision_request,
    process_facet_revision_response,
    form_facet_iterative_request,
    process_facet_iterative_response,
)

from .schema import Facet, AnnotationTable

logger = logger = logging.getLogger(__name__)

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

    def init_facets(
        self,
        task: str,
        videos: Dict[str, List[KnowledgeItem]],
        num_facets: int,
    ) -> List[Facet]:
        """
        Bootstrap the algorithm with common-sense contextual factors.

        Args:
            task: Task name/description
            videos: Dict of video_id -> List of KnowledgeItems
            num_facets: Number of facets to initialize with

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
        )

        return [facet for result in batch_results for facet in result]

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

        target_item_ids_set = set(target_item_ids)

        # Filter to only videos that contain at least one target item — avoids
        # flooding the discovery prompt with irrelevant tutorials.
        target_video_ids = {
            item.source.video_id
            for video_items in videos.values()
            for item in video_items
            if item.id in target_item_ids_set
        }
        target_videos = {vid: items for vid, items in videos.items() if vid in target_video_ids}

        target_items = [
            item
            for video_items in target_videos.values()
            for item in video_items
            if item.id in target_item_ids_set
        ]

        if len(target_items) < 2:
            logger.info("Not enough target_items for discovery")
            return []

        request_args = [{
            "task": task,
            "videos": target_videos,
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

    def propose_facet_iterative(
        self,
        task: str,
        facets: List[Facet],
        context: str,
        videos: Optional[Dict[str, List[KnowledgeItem]]] = None,
        annotations: Optional[AnnotationTable] = None,
        tree_stats: Optional[Dict[str, Any]] = None,
    ) -> List[Facet]:
        """Propose exactly one new facet conditioned on the existing pool.

        Args:
            task: Task name/description
            facets: Current facet pool (definitions + vocabularies shown to LLM)
            context: Substring-checked string of components to include ("facets", "items", "tree")
            videos: Video segments with their pool annotations (included when "items" in context)
            annotations: Current pool annotations for the video segments
            tree_stats: Tree summary dict (included when "tree" in context)

        Returns:
            A list of new Facets, or an empty list if the LLM refused.
        """
        batch_id = f"facet_iterative_{random_uid()}"

        request_args = [{
            "task": task,
            "facets": facets,
            "context": context,
            "videos": videos,
            "annotations": annotations,
            "tree_stats": tree_stats,
        }]

        batch_results = batch_run_llm_calls(
            batch_id,
            request_args,
            form_facet_iterative_request,
            process_facet_iterative_response,
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

        TODO: what items are useful to pass here?
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