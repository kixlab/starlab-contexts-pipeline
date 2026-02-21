"""
Iterative task schema discovery algorithm.

Implements the loop described in pseudo_code.py, following the reference
implementation in server/src/framework_split.py.

The algorithm:
1. Computes a Pareto frontier of schemas (facet subsets) trading off
   discriminativeness vs. compactness via 0/1 knapsack DP.
2. Samples schemas from the frontier and finds collision sites.
3. Proposes new facets to resolve collisions; revises existing ones.
4. Builds vocabulary and labels items for the new facets.
5. Accepts the update if the new frontier Pareto-improves the old one.
6. Repeats until no improvement for max_iterations consecutive rounds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .knowledge_items.schema import KnowledgeBase, KnowledgeItemType
from .knowledge_items.clustering import ClusterConfig

from .facets.schema import Facet, AnnotationTable
from .facets.discovery import VLMFacetExtractor
from .facets.vocabulary import VLMFacetVocabularyBuilder
from .facets.label import VLMFacetLabeler
from .facets.io import (
    get_facets_cache_path,
    get_annotations_cache_path,
    load_facets,
    load_annotations,
    save_facets,
    save_annotations,
)

from .optimization.metrics import compute_cells
from .optimization.frontier import (
    FrontierPoint,
    clip_frontier,
    compute_frontier,
    is_pareto_improved,
    sample_schemas,
)
from .optimization.collision import find_colliding_cells, sample_target_items

logger = logging.getLogger(__name__)


@dataclass
class ExtractionConfig:
    """Knowledge item extraction + clustering settings."""

    granularity: KnowledgeItemType = KnowledgeItemType.ATOMIC
    cluster_config: ClusterConfig = field(default_factory=ClusterConfig)
    provider: Optional[str] = None
    model: Optional[str] = None
    max_concurrency: int = 32
    base_dir: str = "task_rep/static/cache"
    use_cache: bool = True


@dataclass
class ModelConfig:
    """Per-stage VLM provider + model override.

    Both fields fall back to the parent DiscoveryConfig defaults when None.

    Example — use gpt-4o only for facet discovery, mini everywhere else:
        DiscoveryConfig(model="gpt-4o-mini", discovery_model=ModelConfig(model="gpt-4o"))
    """

    provider: Optional[str] = None
    model: Optional[str] = None


@dataclass
class DiscoveryConfig:
    """Configuration for the iterative schema discovery algorithm."""

    # Convergence: stop after this many consecutive iterations with no improvement
    max_iterations: int = 5

    # Frontier sampling
    k_schemas_to_sample: int = 2

    # Only discover new facets from a schema if its D is above this threshold.
    # Schemas already discriminating well are only revised, not expanded.
    target_d: float = 0.8

    # Collision detection
    max_cells_per_iteration: int = 1
    max_items_per_cell: int = 5

    # VLM defaults — used by all stages unless a per-stage override is set below
    task: str = ""
    provider: Optional[str] = None
    model: Optional[str] = None
    max_concurrency: int = 32

    # Per-stage model overrides.  Set only the fields you want to change;
    # unset fields fall back to the global provider/model above.
    discovery_model: Optional[ModelConfig] = None  # VLMFacetExtractor
    vocab_model: Optional[ModelConfig] = None      # VLMFacetVocabularyBuilder
    labeling_model: Optional[ModelConfig] = None   # VLMFacetLabeler

    # Item type filter — when set, only items of this type are used for
    # discovery, vocabulary building, labeling, and metric computation.
    # Run the algorithm once per type to discover type-specific facets.
    # Cache paths are automatically sub-scoped by type value.
    item_type: Optional[KnowledgeItemType] = None

    # Caching — set both to enable per-iteration checkpointing
    base_dir: Optional[str] = None
    task_id: Optional[str] = None
    use_cache: bool = True


@dataclass
class DiscoveryResult:
    """Result of the iterative schema discovery algorithm."""

    facets: List[Facet]
    frontier: List[FrontierPoint]
    annotations: AnnotationTable
    iterations: int


class IterativeSchemaDiscovery:
    """
    Orchestrates iterative task schema discovery.

    Usage:
        config = DiscoveryConfig(task="pandas", base_dir="cache", task_id="v1")
        algo = IterativeSchemaDiscovery(config)
        result = algo.run(knowledge_base)
    """

    def __init__(self, config: DiscoveryConfig):
        self.config = config

        def _provider(override: Optional[ModelConfig]) -> Optional[str]:
            return override.provider if (override and override.provider is not None) else config.provider

        def _model(override: Optional[ModelConfig]) -> Optional[str]:
            return override.model if (override and override.model is not None) else config.model

        self._extractor = VLMFacetExtractor(
            task=config.task,
            provider=_provider(config.discovery_model),
            model=_model(config.discovery_model),
            max_concurrency=config.max_concurrency,
        )
        self._vocab_builder = VLMFacetVocabularyBuilder(
            task=config.task,
            provider=_provider(config.vocab_model),
            model=_model(config.vocab_model),
            max_concurrency=config.max_concurrency,
        )
        self._labeler = VLMFacetLabeler(
            task=config.task,
            provider=_provider(config.labeling_model),
            model=_model(config.labeling_model),
            max_concurrency=config.max_concurrency,
        )

        # Mutable state
        self._facets: Dict[str, Facet] = {}
        self._annotations: AnnotationTable = AnnotationTable()
        self._frontier: List[FrontierPoint] = []
        self._iteration: int = 0

    @property
    def _effective_task_id(self) -> Optional[str]:
        """task_id sub-scoped by item_type when set (e.g. 'sushi_v1/atomic')."""
        if not self.config.task_id:
            return None
        if self.config.item_type:
            return f"{self.config.task_id}/{self.config.item_type.value}"
        return self.config.task_id

    def run(self, knowledge_base: KnowledgeBase) -> DiscoveryResult:
        """
        Run the full iterative schema discovery algorithm.

        Args:
            knowledge_base: Pre-built knowledge base with unit_ids assigned.
                            Items without unit_id are skipped in metrics.
                            If DiscoveryConfig.item_type is set, only items of
                            that type are used; all others are ignored.

        Returns:
            DiscoveryResult with final facets, frontier, and annotations.
        """
        if self.config.item_type is not None:
            knowledge_base = knowledge_base.filter_types([self.config.item_type])
            logger.info("Filtered knowledge base to type '%s': %d items.", self.config.item_type.value, len(knowledge_base))

        self._maybe_restore_cache(knowledge_base)

        self._frontier = compute_frontier(
            knowledge_base,
            list(self._facets.values()),
            self._annotations,
        )
        logger.info(
            "Initial frontier: %d points | best D=%.4f",
            len(self._frontier),
            self._frontier[0].discriminativeness if self._frontier else float("inf"),
        )

        tries_with_no_improvement = 0

        while tries_with_no_improvement < self.config.max_iterations:
            self._iteration += 1
            logger.info("=== Iteration %d ===", self._iteration)

            # Step 1: Sample schemas from the clipped frontier
            clipped = clip_frontier(self._frontier)
            if not clipped:
                logger.info("Frontier fully resolved (all D ≤ min_d); stopping.")
                break

            sampled = sample_schemas(
                clipped,
                self._facets,
                k=self.config.k_schemas_to_sample,
            )
            if not sampled:
                # No existing facets yet — start from empty schema
                sampled = [([], self._frontier[0])]

            # Step 2: Facet generation
            all_new_facets: List[Facet] = []
            videos = knowledge_base.by_video()

            for schema_facets, point in sampled:
                # 2a: Discover new facets from collision sites when D is still high
                if point.discriminativeness > self.config.target_d:
                    cells = compute_cells(
                        knowledge_base, schema_facets, self._annotations
                    )
                    colliding = find_colliding_cells(cells)
                    targets = sample_target_items(
                        knowledge_base,
                        schema_facets,
                        self._annotations,
                        colliding,
                        max_cells=self.config.max_cells_per_iteration,
                        max_items_per_cell=self.config.max_items_per_cell,
                    )
                    for target_ids, cell_sig in targets:
                        proposed = self._extractor.propose_facets(
                            videos=videos,
                            target_item_ids=target_ids,
                            existing_facets=list(self._facets.values()),
                            target_signature=cell_sig,
                        )
                        all_new_facets.extend(proposed)

                # 2b: Revise existing facets in this schema
                if schema_facets:
                    revised = self._extractor.revise_facets(schema_facets)
                    all_new_facets.extend(revised)

            if not all_new_facets:
                logger.info("No new facets discovered/revised.")
                tries_with_no_improvement += 1
                continue

            # Deduplicate by id against the existing pool
            seen_ids = set(self._facets.keys())
            unique_new = []
            for f in all_new_facets:
                if f.id not in seen_ids:
                    seen_ids.add(f.id)
                    unique_new.append(f)

            if not unique_new:
                logger.info("All proposed facets already in pool.")
                tries_with_no_improvement += 1
                continue

            logger.info("Candidate new facets: %d", len(unique_new))

            # Step 3: Build vocabulary for new facets
            unique_new = self._vocab_builder.build_vocabulary(
                unique_new, knowledge_base
            )

            # Step 4: Label items with new facets
            new_annotations = self._labeler.label_batch(unique_new, knowledge_base)

            # Step 5: Tentatively merge and recompute frontier
            combined_facets = list(self._facets.values()) + unique_new
            combined_annotations = self._annotations.merge(new_annotations)
            new_frontier = compute_frontier(
                knowledge_base, combined_facets, combined_annotations
            )

            # Step 6: Accept if the clipped frontier improved
            if is_pareto_improved(self._frontier, new_frontier):
                logger.info("Pareto improvement — accepting %d new facets.", len(unique_new))
                for f in unique_new:
                    self._facets[f.id] = f
                self._annotations.update(new_annotations)
                self._frontier = new_frontier
                tries_with_no_improvement = 0
                self._save_cache()
            else:
                logger.info("No Pareto improvement — discarding candidates.")
                tries_with_no_improvement += 1

        return DiscoveryResult(
            facets=list(self._facets.values()),
            frontier=self._frontier,
            annotations=self._annotations,
            iterations=self._iteration,
        )

    # ---- caching ----

    def _save_cache(self):
        if not self.config.base_dir or not self._effective_task_id:
            return
        save_facets(
            list(self._facets.values()),
            get_facets_cache_path(self.config.base_dir, self._effective_task_id, self._iteration),
        )
        save_annotations(
            self._annotations,
            get_annotations_cache_path(self.config.base_dir, self._effective_task_id, self._iteration),
        )
        logger.info("Saved cache at iteration %d.", self._iteration)

    def _maybe_restore_cache(self, knowledge_base: KnowledgeBase):
        if not self.config.base_dir or not self._effective_task_id or not self.config.use_cache:
            return
        # Find latest iteration with both files present
        for i in range(self.config.max_iterations * 10, 0, -1):
            fp = get_facets_cache_path(self.config.base_dir, self._effective_task_id, i)
            ap = get_annotations_cache_path(self.config.base_dir, self._effective_task_id, i)
            if fp.exists() and ap.exists():
                cached = load_facets(fp)
                self._facets = {f.id: f for f in cached}
                self._annotations = load_annotations(ap)
                self._iteration = i
                logger.info("Restored cache from iteration %d (%d facets).", i, len(cached))
                return
