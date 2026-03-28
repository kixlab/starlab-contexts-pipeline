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
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from pathlib import Path

from utils import ModelConfig

from construct import ConstructKnowledgeBase

from data_prep import (
    Task,
)

from knowledge_items import (
    KnowledgeBase,
    KnowledgeItemType,
    KnowledgeItemSubtype,
    ItemScopeConfig,
    ATOMIC_ALL,
    ATOMIC_SUBTYPE_PRESETS,
)

from facets import (
    Facet,
    AnnotationTable,
    VLMFacetExtractor,
    VLMFacetVocabularyBuilder,
    VLMFacetLabeler,
)

from optimization import (
    compute_cells,
    sample_target_items,
    sample_target_items_coherence,
    sample_schemas,
    compute_irr,

    FrontierPoint,
    clip_frontier,
    compute_frontier,
    is_pareto_improved,

    RefinementTree,
    build_tree_nodes,
    prune_and_connect,
    sample_schemas_from_tree,
    is_tree_improved,
    get_active_facet_ids,
)

logger = logging.getLogger(__name__)

@dataclass
class DiscoveryConfig:
    """Configuration for the iterative schema discovery algorithm."""
    # Per-stage model overrides.  Set only the fields you want to change;
    # unset fields fall back to the global provider/model above.
    discovery_model: ModelConfig  # VLMFacetExtractor
    vocab_model: ModelConfig      # VLMFacetVocabularyBuilder
    labeling_model: ModelConfig   # VLMFacetLabeler
    
    # Item scope — defines which items the algorithm processes.
    # item_scope.item_type selects ATOMIC / STEP / TUTORIAL.
    # item_scope.atomic_subtype restricts *optimization* (frontier / cells / target sampling) to a named preset of atomic subtypes; discovery, vocabulary, and
    # labeling always operate on the full knowledge base for richer context.
    item_scope: ItemScopeConfig

    # Caching — set both to enable per-iteration checkpointing
    use_cache: bool = True

    # Convergence: stop after this many consecutive iterations with no improvement
    max_iterations: int = 5

    # Hard cap on total facets in the pool.  0 = unlimited.
    max_facets: int = 0

    # Frontier sampling
    k_schemas_to_sample: int = 2

    # Only discover new facets from a schema if its D is above this threshold.
    # Schemas already discriminating well are only revised, not expanded.
    target_d: float = 0.8
    
    # Minimum D value for the frontier (i.e., considered resolved)
    min_d: float = 0.01

    # Maximum number of facets in a single schema on the frontier.
    # Brute-force enumeration is O(n^max_schema_facets), so keep this small.
    max_schema_facets: int = 3

    # Collision detection
    max_cells_per_iteration: int = 1
    max_items_per_cell: int = 5

    max_concurrency: int = 32

    # Vocabulary sampling — limits Phase 1 (per-video proposal) to a random
    # sample of this many videos.  `0` means use all videos (original behaviour).
    # Phase 2 consolidation and labeling are always run on the full corpus.
    vocab_sample_videos: int = 0

    # Initialization — bootstrap the iterative algorithm with this many
    # common-sense facets before the main loop.  0 = no initialization.
    init_facets: int = 0

    # Metric mode — which quality metric drives the Pareto frontier.
    # "entropy"             = original discriminativeness (log-based, cross-unit separation)
    # "coherence_ordinal"   = η² using item local_index as position (within-tutorial clustering)
    # "coherence_temporal"  = η² using time_range midpoint as position (within-tutorial clustering)
    metric_mode: str = "entropy"

    # Eta-squared averaging mode (only used when metric_mode is coherence_*).
    # "equal"    = each video contributes equally to the average η²
    # "weighted" = videos weighted by their item count
    eta_squared_weighting: str = "equal"

    # Selection mode — which approach drives schema selection and improvement.
    # "pareto"           = Pareto frontier (discriminativeness vs. compactness)
    # "refinement_tree"  = MI-based refinement tree with greedy pruning
    # "one_shot"         = Non-iterative baseline: bootstrap with init_facets, skip iteration
    selection_mode: str = "pareto"

    # Refinement threshold tau (only used when selection_mode="refinement_tree").
    # A node is considered captured by another if MI(keeper, candidate) / H(candidate) > tau.
    refinement_threshold: float = 0.85

    # Tree complexity cap (only used when selection_mode="refinement_tree").
    # 0 = unlimited; >0 = stop iterating when the tree has >= this many surviving nodes.
    # Useful for matching tree complexity across methods (e.g., set to one_shot's node count).
    max_tree_nodes: int = 0

    # IRR (Inter-Rater Reliability) diagnostic.
    # 0 = disabled; >0 = sample N videos and re-label to measure consistency.
    irr_sample_videos: int = 0
    # Number of independent labeling passes for IRR.
    irr_num_raters: int = 3
    # Minimum Fleiss' kappa for a facet to be retained; facets below this are pruned.
    irr_min_agreement: float = 0.6

    def cache_suffix(self) -> str:
        item_scope = self.item_scope.cache_suffix()
        discovery_model_suffix = self.discovery_model.cache_suffix()
        vocab_model_suffix = self.vocab_model.cache_suffix()
        labeling_model_suffix = self.labeling_model.cache_suffix()
        max_iterations = str(self.max_iterations)
        k_schemas_to_sample = str(self.k_schemas_to_sample)
        target_d = str(self.target_d)
        max_cells_per_iteration = str(self.max_cells_per_iteration)
        max_items_per_cell = str(self.max_items_per_cell)
        vocab_sample_videos = str(self.vocab_sample_videos) if self.vocab_sample_videos != 0 else "all"
        max_schema_facets = str(self.max_schema_facets)
        init_facets = str(self.init_facets) if self.init_facets != 0 else "0"
        metric_mode = self.metric_mode
        eta_squared_weighting = self.eta_squared_weighting if self.metric_mode != "entropy" else "na"
        irr = f"irr{self.irr_sample_videos}x{self.irr_num_raters}a{self.irr_min_agreement}" if self.irr_sample_videos > 0 else "noirr"
        selection = self.selection_mode
        ref_tau = f"tau{self.refinement_threshold}" if self.selection_mode == "refinement_tree" else "na"

        max_facets = str(self.max_facets) if self.max_facets > 0 else "inf"
        max_tree_nodes = f"tn{self.max_tree_nodes}" if self.max_tree_nodes > 0 else "tninf"

        return f"{item_scope}_{discovery_model_suffix}_{vocab_model_suffix}_{labeling_model_suffix}_{max_iterations}_{k_schemas_to_sample}_{target_d}_{max_cells_per_iteration}_{max_items_per_cell}_{vocab_sample_videos}_{max_schema_facets}_{init_facets}_{metric_mode}_{eta_squared_weighting}_{irr}_{selection}_{ref_tau}_{max_facets}_{max_tree_nodes}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_scope": self.item_scope.to_dict(),
            "use_cache": self.use_cache,
            "max_iterations": self.max_iterations,
            "max_facets": self.max_facets,
            "k_schemas_to_sample": self.k_schemas_to_sample,
            "target_d": self.target_d,
            "min_d": self.min_d,
            "max_cells_per_iteration": self.max_cells_per_iteration,
            "max_items_per_cell": self.max_items_per_cell,
            "max_concurrency": self.max_concurrency,
            "vocab_sample_videos": self.vocab_sample_videos,
            "max_schema_facets": self.max_schema_facets,
            "init_facets": self.init_facets,
            "metric_mode": self.metric_mode,
            "eta_squared_weighting": self.eta_squared_weighting,
            "selection_mode": self.selection_mode,
            "refinement_threshold": self.refinement_threshold,
            "max_tree_nodes": self.max_tree_nodes,
            "irr_sample_videos": self.irr_sample_videos,
            "irr_num_raters": self.irr_num_raters,
            "irr_min_agreement": self.irr_min_agreement,
            "discovery_model": self.discovery_model.to_dict(),
            "vocab_model": self.vocab_model.to_dict(),
            "labeling_model": self.labeling_model.to_dict(),
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "DiscoveryConfig":
        return DiscoveryConfig(
            item_scope=ItemScopeConfig.from_dict(data["item_scope"]),
            discovery_model=ModelConfig.from_dict(data["discovery_model"]),
            vocab_model=ModelConfig.from_dict(data["vocab_model"]),
            labeling_model=ModelConfig.from_dict(data["labeling_model"]),
            use_cache=data.get("use_cache", True),
            max_iterations=data.get("max_iterations", 5),
            max_facets=data.get("max_facets", 0),
            k_schemas_to_sample=data.get("k_schemas_to_sample", 2),
            target_d=data.get("target_d", 0.8),
            min_d=data.get("min_d", 0.01),
            max_cells_per_iteration=data.get("max_cells_per_iteration", 1),
            max_items_per_cell=data.get("max_items_per_cell", 5),
            max_concurrency=data.get("max_concurrency", 32),
            vocab_sample_videos=data.get("vocab_sample_videos", 0),
            max_schema_facets=data.get("max_schema_facets", 3),
            init_facets=data.get("init_facets", 0),
            metric_mode=data.get("metric_mode", "entropy"),
            eta_squared_weighting=data.get("eta_squared_weighting", "equal"),
            selection_mode=data.get("selection_mode", "pareto"),
            refinement_threshold=data.get("refinement_threshold", 0.85),
            max_tree_nodes=data.get("max_tree_nodes", 0),
            irr_sample_videos=data.get("irr_sample_videos", 0),
            irr_num_raters=data.get("irr_num_raters", 3),
            irr_min_agreement=data.get("irr_min_agreement", 0.6),
        )


@dataclass
class DiscoveryResult:
    """Result of the iterative schema discovery algorithm."""

    facets: List[Facet]
    frontier: List[FrontierPoint]
    annotations: AnnotationTable
    knowledge_base: KnowledgeBase
    iterations: int
    tree: Optional[RefinementTree] = None  # populated in refinement_tree mode
    discovery_calls: int = 0  # total VLM discovery/revision calls (init + propose + revise)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "facets": [f.to_dict() for f in self.facets],
            "frontier": [p.to_dict() for p in self.frontier],
            "annotations": self.annotations.to_list(),
            "knowledge_base": self.knowledge_base.to_dict(),
            "iterations": self.iterations,
            "discovery_calls": self.discovery_calls,
        }
        if self.tree is not None:
            d["tree"] = self.tree.to_dict()
        return d

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> DiscoveryResult:
        tree = None
        if "tree" in data and data["tree"] is not None:
            tree = RefinementTree.from_dict(data["tree"])
        return DiscoveryResult(
            facets=[Facet.from_dict(f) for f in data["facets"]],
            frontier=[FrontierPoint.from_dict(p) for p in data["frontier"]],
            annotations=AnnotationTable.from_list(data["annotations"]),
            knowledge_base=KnowledgeBase.from_dict(data["knowledge_base"]),
            iterations=data["iterations"],
            tree=tree,
            discovery_calls=data.get("discovery_calls", 0),
        )

class IterativeSchemaDiscovery:
    """
    Orchestrates iterative task schema discovery.

    Usage:
        config = DiscoveryConfig(task="pandas")
        algo = IterativeSchemaDiscovery(config)
        result = algo.run()
    """

    def __init__(self, config: DiscoveryConfig):
        self.config = config

        self._extractor = VLMFacetExtractor(
            model_config=self.config.discovery_model,
            max_concurrency=self.config.max_concurrency,
        )
        self._vocab_builder = VLMFacetVocabularyBuilder(
            model_config=self.config.vocab_model,
            max_concurrency=self.config.max_concurrency,
        )
        self._labeler = VLMFacetLabeler(
            model_config=self.config.labeling_model,
            max_concurrency=self.config.max_concurrency,
        )
        # Mutable state
        self._facets: Dict[str, Facet] = {}
        self._annotations: AnnotationTable = AnnotationTable()
        self._frontier: List[FrontierPoint] = []
        self._tree: Optional[RefinementTree] = None
        self._knowledge_base: KnowledgeBase = KnowledgeBase(items=[])
        self._iteration: int = 0
        self._discovery_calls: int = 0

    def get_result_path(self, cache_dir: str | Path) -> Path:
        return cache_dir / f"run_result_{self.config.cache_suffix()}.json"

    def run(self, task: Task, constructor: ConstructKnowledgeBase) -> DiscoveryResult:
        """
        Run the full iterative schema discovery algorithm.

        Args:
            task: Task object.
            constructor: ConstructKnowledgeBase instance.

        Returns:
            DiscoveryResult with final facets, frontier, annotations, and knowledge base.
        """
        use_cache = self.config.use_cache
        item_type = self.config.item_scope.item_type
        cache_dir = constructor.get_cache_dir(task)
        self._knowledge_base = constructor.get_knowledge_base(task, item_type)
        logger.info("Knowledge base: %d items of type '%s'.", len(self._knowledge_base), item_type.value)

        # Focused KB for optimization (frontier / cells / target sampling).
        # Discovery, vocab, and labeling always use the full self._knowledge_base.
        focused_kb = self._knowledge_base
        subtype_filter = self.config.item_scope.atomic_subtype
        if subtype_filter.subtypes:
            focused_kb = self._knowledge_base.filter_subtypes(subtype_filter.subtypes)
            logger.info(
                "Subtype focus '%s': %d/%d items in optimization scope.",
                subtype_filter.name, len(focused_kb), len(self._knowledge_base),
            )

        if use_cache:
            self._restore_cache(cache_dir)

        init_facets_count = 0

        # one_shot mode: use max_facets as the bootstrap count.
        if self.config.selection_mode == "one_shot":
            if self.config.max_facets <= 0:
                raise ValueError(
                    "one_shot mode requires max_facets > 0 to determine "
                    "how many facets to bootstrap."
                )
            init_facets_count = self.config.max_facets
            logger.info(
                "one_shot mode: bootstrapping %d facet(s), no iteration.",
                init_facets_count,
            )

        if init_facets_count == 0:
            init_facets_count = self.config.init_facets
        # Optional: bootstrap with common-sense facets before iterating.
        # Only runs when init_facets > 0 and no facets exist yet (fresh run).
        if init_facets_count > 0 and not self._facets:
            logger.info(
                "Initializing with %d common-sense facet(s)...",
                init_facets_count,
            )
            videos = self._knowledge_base.by_video()
            init_facets = self._extractor.init_facets(
                task=task.name,
                videos=videos,
                num_facets=init_facets_count,
            )
            self._discovery_calls += 1
            # IRR check — signal unreliable facets
            if self.config.irr_sample_videos > 0 and init_facets:
                irr = compute_irr(
                    task=task.name,
                    facets=init_facets,
                    knowledge_base=self._knowledge_base,
                    labeler=self._labeler,
                    sample_videos=self.config.irr_sample_videos,
                    num_raters=self.config.irr_num_raters,
                    min_agreement=self.config.irr_min_agreement,
                )
                low_irr_ids = set(irr.get("low_irr_facet_ids", []))
                logger.info(
                    "There are %d low-IRR candidate(s) among init facets: %s",
                    len(low_irr_ids), list(low_irr_ids),
                )

            if init_facets:
                init_facets = self._vocab_builder.build_vocabulary(
                    task=task.name,
                    facets=init_facets,
                    knowledge_base=self._knowledge_base,
                    sample_videos=self.config.vocab_sample_videos,
                )
                init_annotations = self._labeler.label_batch(
                    task=task.name,
                    facets=init_facets,
                    knowledge_base=self._knowledge_base,
                )
                for f in init_facets:
                    self._facets[f.id] = f
                self._annotations.update(init_annotations)
                logger.info("Initialized %d facet(s).", len(init_facets))
                self._save_cache(cache_dir)

        max_schema_facets = self.config.max_schema_facets
        metric_mode = self.config.metric_mode
        eta_squared_weighting = self.config.eta_squared_weighting
        selection_mode = self.config.selection_mode

        if selection_mode == "refinement_tree":
            nodes, h_memo = build_tree_nodes(
                focused_kb, list(self._facets.values()), self._annotations,
                max_schema_facets=max_schema_facets,
                metric_mode=metric_mode,
                eta_squared_weighting=eta_squared_weighting,
            )
            self._tree = prune_and_connect(
                nodes, focused_kb, self._facets, self._annotations, h_memo,
                refinement_threshold=self.config.refinement_threshold,
                metric_mode=metric_mode,
            )
            logger.info(
                "Initial tree: %d surviving nodes | %d edges",
                len(self._tree.nodes), len(self._tree.edges),
            )
        else:
            self._frontier = compute_frontier(
                focused_kb,
                list(self._facets.values()),
                self._annotations,
                max_schema_facets=max_schema_facets,
                metric_mode=metric_mode,
                eta_squared_weighting=eta_squared_weighting,
            )
            logger.info(
                "Initial frontier: %d points | worst D=%.4f | best D=%.4f",
                len(self._frontier),
                self._frontier[0].discriminativeness if self._frontier else float("inf"),
                self._frontier[-1].discriminativeness if self._frontier else float("inf"),
            )

        tries_with_no_improvement = 0
        target_d = self.config.target_d
        max_iterations = self.config.max_iterations
        max_facets = self.config.max_facets
        max_tree_nodes = self.config.max_tree_nodes
        max_cells_per_iteration = self.config.max_cells_per_iteration
        max_items_per_cell = self.config.max_items_per_cell
        k_schemas_to_sample = self.config.k_schemas_to_sample
        vocab_sample_videos = self.config.vocab_sample_videos
        min_d = self.config.min_d

        while selection_mode != "one_shot" and tries_with_no_improvement < max_iterations:
            self._iteration += 1
            logger.info("=== Iteration %d ===", self._iteration)

            # Print distribution of cell sizes
            cells = compute_cells(
                focused_kb, list(self._facets.values()), self._annotations
            )
            size_counts = defaultdict(int)
            for _, units in cells.items():
                size_counts[len(units)] += 1
            total_cell_sizes = sum(size_counts.values())
            logger.info("Cell size distribution (total=%d units):", total_cell_sizes)
            for size, count in size_counts.items():
                logger.info(f"{size} units: {count} ({count/total_cell_sizes*100:.1f}%)")

            # Step 1: Sample schemas for expansion
            if selection_mode == "refinement_tree":
                sampled = sample_schemas_from_tree(
                    self._tree,
                    self._facets,
                    target_d=target_d,
                    k=k_schemas_to_sample,
                    metric_mode=metric_mode,
                )
                if not sampled:
                    logger.info("Tree fully resolved (no unresolved leaves); stopping.")
                    break
            else:
                clipped = clip_frontier(self._frontier, min_d=target_d)
                if not clipped:
                    logger.info("Frontier fully resolved (all D ≤ %s); stopping.", self.config.target_d)
                    break
                sampled = sample_schemas(
                    clipped,
                    self._facets,
                    k=k_schemas_to_sample,
                )
                assert sampled, "No schemas sampled from the frontier."

            # Step 2: Facet generation
            all_new_facets: List[Facet] = []
            videos = self._knowledge_base.by_video()

            for schema_facets, point in sampled:
                # 2a: Discover new facets from collision sites when D is still high
                if point.discriminativeness > target_d:
                    if metric_mode == "entropy":
                        targets = sample_target_items(
                            focused_kb,
                            schema_facets,
                            self._annotations,
                            max_cells=max_cells_per_iteration,
                            max_items_per_cell=max_items_per_cell,
                        )
                    else:
                        pos_mode = "ordinal" if metric_mode == "coherence_ordinal" else "time"
                        targets = sample_target_items_coherence(
                            focused_kb,
                            schema_facets,
                            self._annotations,
                            position_mode=pos_mode,
                            max_cells=max_cells_per_iteration,
                        )
                    for target_ids, cell_sig in targets:
                        proposed = self._extractor.propose_facets(
                            task=task.name,
                            videos=videos,
                            target_item_ids=target_ids,
                            facets=schema_facets,
                            target_signature=cell_sig,
                        )
                        self._discovery_calls += 1
                        all_new_facets.extend(proposed)

                # 2b: Revise existing facets in this schema
                if schema_facets:
                    revised = self._extractor.revise_facets(
                        task=task.name,
                        facets=schema_facets
                    )
                    self._discovery_calls += 1
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

            # Step 3.5: IRR check — prune unreliable facets before full annotation
            if self.config.irr_sample_videos > 0 and unique_new:
                irr = compute_irr(
                    task=task.name,
                    facets=unique_new,
                    knowledge_base=self._knowledge_base,
                    labeler=self._labeler,
                    sample_videos=self.config.irr_sample_videos,
                    num_raters=self.config.irr_num_raters,
                    min_agreement=self.config.irr_min_agreement,
                )
                low_irr_ids = set(irr.get("low_irr_facet_ids", []))
                if low_irr_ids:
                    unique_new = [f for f in unique_new if f.id not in low_irr_ids]
                    logger.info(
                        "Pruned %d low-IRR candidate(s): %s",
                        len(low_irr_ids), list(low_irr_ids),
                    )
                if not unique_new:
                    logger.info("All candidates pruned by IRR.")
                    tries_with_no_improvement += 1
                    continue
            
            # Step 3: Build vocabulary for new facets (TODO: should put before irr?)
            unique_new = self._vocab_builder.build_vocabulary(
                task=task.name,
                facets=unique_new,
                knowledge_base=self._knowledge_base,
                sample_videos=vocab_sample_videos,
            )

            # Step 4: Label items with new facets
            new_annotations = self._labeler.label_batch(
                task=task.name,
                facets=unique_new,
                knowledge_base=self._knowledge_base,
            )

            # Step 5: Tentatively merge and recompute selection
            combined_facets = list(self._facets.values()) + unique_new
            combined_annotations = self._annotations.merge(new_annotations)

            if selection_mode == "refinement_tree":
                new_nodes, new_h_memo = build_tree_nodes(
                    focused_kb, combined_facets, combined_annotations,
                    max_schema_facets=max_schema_facets,
                    metric_mode=metric_mode,
                    eta_squared_weighting=eta_squared_weighting,
                )
                new_tree = prune_and_connect(
                    new_nodes, focused_kb,
                    {f.id: f for f in combined_facets},
                    combined_annotations, new_h_memo,
                    refinement_threshold=self.config.refinement_threshold,
                    metric_mode=metric_mode,
                )
                improved = is_tree_improved(self._tree, new_tree, metric_mode=metric_mode)
            else:
                new_frontier = compute_frontier(
                    focused_kb, combined_facets, combined_annotations,
                    max_schema_facets=max_schema_facets,
                    metric_mode=metric_mode,
                    eta_squared_weighting=eta_squared_weighting,
                )
                improved = is_pareto_improved(self._frontier, new_frontier, min_d=target_d)

            # Step 6: Accept if improved
            if improved:
                if selection_mode == "refinement_tree":
                    logger.info("-------------------------------- New tree --------------------------------")
                    for node in new_tree.nodes:
                        logger.info(f"Node: {node.id} | size={node.size} D={node.D:.4f} eta2={node.eta2:.4f}")
                    logger.info("---------------------------------------------------------------------")

                    active_ids = get_active_facet_ids(new_tree, min_d=min_d, metric_mode=metric_mode)
                else:
                    logger.info("-------------------------------- New frontier --------------------------------")
                    for point in new_frontier:
                        logger.info(f"New: {point.discriminativeness} {point.compactness}")
                    logger.info("---------------------------------------------------------------------")

                    active_ids: set[str] = set()
                    for point in clip_frontier(new_frontier, min_d=min_d):
                        active_ids.update(point.facet_ids)

                accepted_new = [f for f in unique_new if f.id in active_ids]
                n_dropped = len(unique_new) - len(accepted_new)
                logger.info(
                    "Improvement — accepting %d new facet(s)%s.",
                    len(accepted_new),
                    f" ({n_dropped} dropped as non-contributing)" if n_dropped else "",
                )

                for f in accepted_new:
                    self._facets[f.id] = f
                self._annotations.update(new_annotations.filter_facets(active_ids))

                # Recompute from the accepted pool
                if selection_mode == "refinement_tree":
                    acc_nodes, acc_h_memo = build_tree_nodes(
                        focused_kb, list(self._facets.values()), self._annotations,
                        max_schema_facets=max_schema_facets,
                        metric_mode=metric_mode,
                        eta_squared_weighting=eta_squared_weighting,
                    )
                    self._tree = prune_and_connect(
                        acc_nodes, focused_kb, self._facets, self._annotations, acc_h_memo,
                        refinement_threshold=self.config.refinement_threshold,
                        metric_mode=metric_mode,
                    )
                else:
                    self._frontier = compute_frontier(
                        focused_kb, list(self._facets.values()), self._annotations,
                        max_schema_facets=max_schema_facets,
                        metric_mode=metric_mode,
                        eta_squared_weighting=eta_squared_weighting,
                    )

                tries_with_no_improvement = 0
                self._save_cache(cache_dir)

                if max_facets > 0 and len(self._facets) >= max_facets:
                    logger.info(
                        "Facet pool reached cap (%d/%d) — stopping.",
                        len(self._facets), max_facets,
                    )
                    break

                if (selection_mode == "refinement_tree"
                    and max_tree_nodes > 0
                    and self._tree
                    and len(self._tree.nodes) >= max_tree_nodes):
                    logger.info(
                        "Tree complexity reached cap (%d/%d surviving nodes) — stopping.",
                        len(self._tree.nodes), max_tree_nodes,
                    )
                    break
            else:
                logger.info("No improvement — discarding candidates.")
                tries_with_no_improvement += 1

        if selection_mode == "one_shot":
            logger.info(
                "one_shot baseline complete: %d facet(s), %d frontier point(s).",
                len(self._facets), len(self._frontier),
            )

        # Unified evaluation: always build a refinement tree from the final pool
        # so that all selection modes produce comparable tree metrics.
        if self._tree is None and self._facets:
            final_nodes, final_h_memo = build_tree_nodes(
                focused_kb, list(self._facets.values()), self._annotations,
                max_schema_facets=max_schema_facets,
                metric_mode=metric_mode,
                eta_squared_weighting=eta_squared_weighting,
            )
            self._tree = prune_and_connect(
                final_nodes, focused_kb, self._facets, self._annotations, final_h_memo,
                refinement_threshold=self.config.refinement_threshold,
                metric_mode=metric_mode,
            )
            logger.info(
                "Unified eval tree: %d surviving nodes | %d edges",
                len(self._tree.nodes), len(self._tree.edges),
            )

        self._save_cache(cache_dir)
        return DiscoveryResult(
            facets=list(self._facets.values()),
            frontier=self._frontier,
            annotations=self._annotations,
            knowledge_base=self._knowledge_base,
            iterations=self._iteration,
            tree=self._tree,
            discovery_calls=self._discovery_calls,
        )

    # ---- caching ----

    def _save_cache(self, cache_dir: str | Path):
        result_path = self.get_result_path(cache_dir)

        result = DiscoveryResult(
            facets=list(self._facets.values()),
            frontier=self._frontier,
            annotations=self._annotations,
            knowledge_base=self._knowledge_base,
            iterations=self._iteration,
            tree=self._tree,
            discovery_calls=self._discovery_calls,
        )

        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
        
        logger.info("Saved cache to %s.", result_path)

    def _restore_cache(self, cache_dir: str | Path):
        result_path = self.get_result_path(cache_dir)
        if result_path.exists():
            with open(result_path, "r", encoding="utf-8") as f:
                result = DiscoveryResult.from_dict(json.load(f))
                ### compare knowledge bases to decide if we need to restore
                if result.knowledge_base != self._knowledge_base:
                    logger.info("Knowledge base mismatch — restoring from scratch.")
                    return
                self._knowledge_base = result.knowledge_base
                self._facets = {f.id: f for f in result.facets}
                self._annotations = result.annotations
                logger.info("Restored cache from %s (%d facets, %d annotations).", result_path, len(result.facets), len(result.annotations))
        else:
            logger.info("No cache found at %s.", result_path)
