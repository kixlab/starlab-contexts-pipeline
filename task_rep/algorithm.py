"""
Iterative task schema discovery algorithm.

The algorithm finds the best pool of `target_facets` facets that produces
the highest-quality refinement tree, within a discovery budget of `max_facets`.

Two selection modes share the same greedy pool-building logic:
- one_shot: generates max_facets facets at once, then greedily selects best pool
- refinement_tree: proposes facets iteratively, maintaining best pool via greedy replacement

Tree quality: more surviving nodes = better. Ties broken by fewer unique facets.

Greedy pool building:
- If pool < target_facets: add candidate
- If pool >= target_facets: try replacing each member, keep best improvement
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
    compute_irr,

    RefinementTree,
    build_tree_nodes,
    prune_and_connect,
    sample_schemas_from_tree,
    compare_tree_quality,
)

logger = logging.getLogger(__name__)

@dataclass
class DiscoveryConfig:
    """Configuration for the iterative schema discovery algorithm."""
    # Per-stage model overrides.
    discovery_model: ModelConfig  # VLMFacetExtractor
    vocab_model: ModelConfig      # VLMFacetVocabularyBuilder
    labeling_model: ModelConfig   # VLMFacetLabeler

    # Item scope — defines which items the algorithm processes.
    item_scope: ItemScopeConfig

    # Caching
    use_cache: bool = True

    # Convergence: stop after this many consecutive iterations with no improvement
    max_iterations: int = 5

    # Discovery budget: total facets proposed (accepted or not). Required > 0.
    max_facets: int = 0

    # Desired pool size. Required > 0.
    target_facets: int = 0

    # Frontier sampling
    k_schemas_to_sample: int = 2

    # Maximum number of facets in a single schema on the frontier.
    # Brute-force enumeration is O(n^max_schema_facets), so keep this small.
    max_schema_facets: int = 3

    # Collision detection
    max_cells_per_iteration: int = 1
    max_items_per_cell: int = 5

    max_concurrency: int = 32

    # Vocabulary sampling — limits Phase 1 (per-video proposal) to a random
    # sample of this many videos.  0 = use all videos.
    vocab_sample_videos: int = 0

    # Initialization — bootstrap the iterative algorithm with this many
    # common-sense facets before the main loop.  0 = no initialization.
    init_facets: int = 0

    # Metric mode — which quality metric drives the tree.
    # "entropy"             = discriminativeness (log-based, cross-unit separation)
    # "coherence_ordinal"   = η² using item local_index as position
    # "coherence_temporal"  = η² using time_range midpoint as position
    metric_mode: str = "entropy"

    # Eta-squared averaging mode (only used when metric_mode is coherence_*).
    eta_squared_weighting: str = "equal"

    # Selection mode — "one_shot" or "refinement_tree"
    selection_mode: str = "refinement_tree"

    # Refinement threshold tau for tree pruning.
    refinement_threshold: float = 0.85

    # IRR (Inter-Rater Reliability) diagnostic.
    irr_sample_videos: int = 0
    irr_num_raters: int = 3
    irr_min_agreement: float = 0.6

    def cache_suffix(self) -> str:
        item_scope = self.item_scope.cache_suffix()
        discovery_model_suffix = self.discovery_model.cache_suffix()
        vocab_model_suffix = self.vocab_model.cache_suffix()
        labeling_model_suffix = self.labeling_model.cache_suffix()
        max_iterations = str(self.max_iterations)
        k_schemas_to_sample = str(self.k_schemas_to_sample)
        max_cells_per_iteration = str(self.max_cells_per_iteration)
        max_items_per_cell = str(self.max_items_per_cell)
        vocab_sample_videos = str(self.vocab_sample_videos) if self.vocab_sample_videos != 0 else "all"
        max_schema_facets = str(self.max_schema_facets)
        init_facets = str(self.init_facets) if self.init_facets != 0 else "0"
        metric_mode = self.metric_mode
        eta_squared_weighting = self.eta_squared_weighting if self.metric_mode != "entropy" else "na"
        irr = f"irr{self.irr_sample_videos}x{self.irr_num_raters}a{self.irr_min_agreement}" if self.irr_sample_videos > 0 else "noirr"
        selection = self.selection_mode
        ref_tau = f"tau{self.refinement_threshold}"

        max_facets = str(self.max_facets) if self.max_facets > 0 else "inf"
        target_facets = str(self.target_facets) if self.target_facets > 0 else "inf"

        return f"{item_scope}_{discovery_model_suffix}_{vocab_model_suffix}_{labeling_model_suffix}_{max_iterations}_{k_schemas_to_sample}_{max_cells_per_iteration}_{max_items_per_cell}_{vocab_sample_videos}_{max_schema_facets}_{init_facets}_{metric_mode}_{eta_squared_weighting}_{irr}_{selection}_{ref_tau}_{max_facets}_{target_facets}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_scope": self.item_scope.to_dict(),
            "use_cache": self.use_cache,
            "max_iterations": self.max_iterations,
            "max_facets": self.max_facets,
            "target_facets": self.target_facets,
            "k_schemas_to_sample": self.k_schemas_to_sample,
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
            target_facets=data.get("target_facets", 0),
            k_schemas_to_sample=data.get("k_schemas_to_sample", 2),
            max_cells_per_iteration=data.get("max_cells_per_iteration", 1),
            max_items_per_cell=data.get("max_items_per_cell", 5),
            max_concurrency=data.get("max_concurrency", 32),
            vocab_sample_videos=data.get("vocab_sample_videos", 0),
            max_schema_facets=data.get("max_schema_facets", 3),
            init_facets=data.get("init_facets", 0),
            metric_mode=data.get("metric_mode", "entropy"),
            eta_squared_weighting=data.get("eta_squared_weighting", "equal"),
            selection_mode=data.get("selection_mode", "refinement_tree"),
            refinement_threshold=data.get("refinement_threshold", 0.85),
            irr_sample_videos=data.get("irr_sample_videos", 0),
            irr_num_raters=data.get("irr_num_raters", 3),
            irr_min_agreement=data.get("irr_min_agreement", 0.6),
        )


@dataclass
class DiscoveryResult:
    """Result of the iterative schema discovery algorithm."""

    facets: List[Facet]
    annotations: AnnotationTable
    knowledge_base: KnowledgeBase
    iterations: int
    tree: Optional[RefinementTree] = None
    discovery_calls: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "facets": [f.to_dict() for f in self.facets],
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
        config = DiscoveryConfig(...)
        algo = IterativeSchemaDiscovery(config)
        result = algo.run(task, constructor)
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
        # Mutable state — current best pool
        self._facets: Dict[str, Facet] = {}
        self._annotations: AnnotationTable = AnnotationTable()
        self._tree: Optional[RefinementTree] = None
        self._knowledge_base: KnowledgeBase = KnowledgeBase(items=[])
        self._iteration: int = 0
        self._discovery_calls: int = 0
        # Facet bank — ALL discovered facets with vocab+annotations (append-only)
        self._bank_facets: Dict[str, Facet] = {}
        self._bank_annotations: AnnotationTable = AnnotationTable()

    def get_result_path(self, cache_dir: str | Path) -> Path:
        return cache_dir / f"run_result_{self.config.cache_suffix()}.json"

    def _get_facet_bank_path(self, cache_dir: str | Path) -> Path:
        """Cache path for the facet bank.

        Keyed only by models + item scope so that facets can be reused across
        different target_facets / max_facets / max_iterations / selection_mode.

        **Important caveat**: facets in the bank were discovered under specific
        algorithm parameters (metric_mode, refinement_threshold, k_schemas_to_sample,
        etc.) that influence which schemas are sampled and which collisions are
        found. Reusing the bank with different parameters is a practical shortcut
        — the vocab and labels are still valid — but the *set of facets* may not
        be the same set that would have been discovered under the new parameters.
        The bank file stores the discovery config of each contributing run so
        this provenance is visible.
        """
        item_scope = self.config.item_scope.cache_suffix()
        discovery = self.config.discovery_model.cache_suffix()
        vocab = self.config.vocab_model.cache_suffix()
        labeling = self.config.labeling_model.cache_suffix()
        return Path(cache_dir) / f"facet_bank_{item_scope}_{discovery}_{vocab}_{labeling}.json"

    def _save_facet_bank(self, cache_dir: str | Path):
        path = self._get_facet_bank_path(cache_dir)
        # Record this run's discovery parameters for provenance
        run_config = {
            "selection_mode": self.config.selection_mode,
            "metric_mode": self.config.metric_mode,
            "refinement_threshold": self.config.refinement_threshold,
            "max_schema_facets": self.config.max_schema_facets,
            "k_schemas_to_sample": self.config.k_schemas_to_sample,
            "max_cells_per_iteration": self.config.max_cells_per_iteration,
            "max_items_per_cell": self.config.max_items_per_cell,
        }
        # Load existing provenance if present
        existing_provenance = []
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    old = json.load(f)
                existing_provenance = old.get("discovery_provenance", [])
            except (json.JSONDecodeError, KeyError):
                pass
        # Append if this config is new
        if run_config not in existing_provenance:
            existing_provenance.append(run_config)

        data = {
            "facets": [f.to_dict() for f in self._bank_facets.values()],
            "annotations": self._bank_annotations.to_list(),
            "discovery_provenance": existing_provenance,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("Saved facet bank to %s (%d facets, %d provenance entries).",
                     path, len(self._bank_facets), len(existing_provenance))

    def _load_facet_bank(self, cache_dir: str | Path):
        path = self._get_facet_bank_path(cache_dir)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            facets = [Facet.from_dict(d) for d in data["facets"]]
            self._bank_facets = {f.id: f for f in facets}
            self._bank_annotations = AnnotationTable.from_list(data["annotations"])
            provenance = data.get("discovery_provenance", [])
            logger.info("Loaded facet bank from %s (%d facets).", path, len(self._bank_facets))
            if provenance:
                logger.info("Bank provenance — discovered under %d config(s):", len(provenance))
                for p in provenance:
                    logger.info("  %s", p)

    def _add_to_bank(self, facets: List[Facet], annotations: AnnotationTable):
        """Add facets and their annotations to the bank."""
        for f in facets:
            self._bank_facets[f.id] = f
        self._bank_annotations.update(annotations)

    def _build_tree(
        self,
        focused_kb: KnowledgeBase,
        facets: List[Facet],
        annotations: AnnotationTable,
    ) -> RefinementTree:
        """Build a refinement tree from the given facets and annotations."""
        nodes, h_memo = build_tree_nodes(
            focused_kb, facets, annotations,
            max_schema_facets=self.config.max_schema_facets,
            metric_mode=self.config.metric_mode,
            eta_squared_weighting=self.config.eta_squared_weighting,
        )
        return prune_and_connect(
            nodes, focused_kb,
            {f.id: f for f in facets},
            annotations, h_memo,
            refinement_threshold=self.config.refinement_threshold,
            metric_mode=self.config.metric_mode,
        )

    def _greedy_select_pool(
        self,
        candidates: List[Facet],
        focused_kb: KnowledgeBase,
        new_annotations: AnnotationTable,
    ) -> bool:
        """
        Try to improve pool by adding/replacing with each candidate.

        - If pool < target_facets: add candidate directly
        - If pool >= target_facets: try replacing each member, keep best

        Returns True if pool changed.
        """
        pool_changed = False
        current_pool = list(self._facets.values())
        target = self.config.target_facets

        for candidate in candidates:
            if len(current_pool) < target:
                # Phase 1: just add until we reach target_facets
                current_pool.append(candidate)
                self._facets = {f.id: f for f in current_pool}
                self._annotations = self._annotations.merge(new_annotations).filter_facets(
                    list(self._facets.keys())
                )
                self._tree = self._build_tree(focused_kb, current_pool, self._annotations)
                pool_changed = True
                logger.info(
                    "Added facet %s to pool (%d/%d).",
                    candidate.id, len(current_pool), target,
                )
            else:
                # Phase 2: try replacing each member, keep best
                best_tree = self._tree  # baseline
                best_idx = -1

                for i in range(len(current_pool)):
                    trial_pool = current_pool[:i] + [candidate] + current_pool[i+1:]
                    trial_ids = [f.id for f in trial_pool]
                    trial_ann = self._annotations.merge(new_annotations).filter_facets(trial_ids)
                    trial_tree = self._build_tree(focused_kb, trial_pool, trial_ann)

                    if compare_tree_quality(trial_tree, best_tree) > 0:
                        best_tree = trial_tree
                        best_idx = i

                if best_idx >= 0:
                    replaced = current_pool[best_idx]
                    current_pool[best_idx] = candidate
                    self._facets = {f.id: f for f in current_pool}
                    self._annotations = self._annotations.merge(new_annotations).filter_facets(
                        list(self._facets.keys())
                    )
                    self._tree = best_tree
                    pool_changed = True
                    logger.info(
                        "Replaced facet %s with %s (tree: %d nodes).",
                        replaced.id, candidate.id, len(best_tree.nodes),
                    )

        return pool_changed

    def run(self, task: Task, constructor: ConstructKnowledgeBase) -> DiscoveryResult:
        """
        Run the full iterative schema discovery algorithm.

        Args:
            task: Task object.
            constructor: ConstructKnowledgeBase instance.

        Returns:
            DiscoveryResult with final facets, annotations, and knowledge base.
        """
        if self.config.max_facets <= 0:
            raise ValueError("max_facets must be > 0 (discovery budget).")
        if self.config.target_facets <= 0:
            raise ValueError("target_facets must be > 0 (desired pool size).")

        use_cache = self.config.use_cache
        item_type = self.config.item_scope.item_type
        cache_dir = constructor.get_cache_dir(task)
        self._knowledge_base = constructor.get_knowledge_base(task, item_type)
        logger.info("Knowledge base: %d items of type '%s'.", len(self._knowledge_base), item_type.value)

        # Focused KB for optimization (tree / cells / target sampling).
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
            self._load_facet_bank(cache_dir)
            self._restore_cache(cache_dir)

        selection_mode = self.config.selection_mode

        if selection_mode == "one_shot":
            self._run_one_shot(task, focused_kb, cache_dir)
        elif selection_mode == "refinement_tree":
            self._run_refinement_tree(task, focused_kb, cache_dir)
        else:
            raise ValueError(f"Unknown selection_mode: {selection_mode!r}. Use 'one_shot' or 'refinement_tree'.")

        self._save_cache(cache_dir)
        return DiscoveryResult(
            facets=list(self._facets.values()),
            annotations=self._annotations,
            knowledge_base=self._knowledge_base,
            iterations=self._iteration,
            tree=self._tree,
            discovery_calls=self._discovery_calls,
        )

    def _run_one_shot(self, task: Task, focused_kb: KnowledgeBase, cache_dir: str | Path):
        """
        one_shot: generate max_facets at once, then greedily select best target_facets pool.

        Uses the facet bank: if the bank already has >= max_facets facets,
        skip generation entirely and just re-run greedy selection.
        """
        max_facets = self.config.max_facets

        if not self._facets:
            # Use bank if it already has enough facets
            if len(self._bank_facets) >= max_facets:
                logger.info(
                    "Facet bank has %d facets (need %d) — skipping generation.",
                    len(self._bank_facets), max_facets,
                )
                all_facets = list(self._bank_facets.values())[:max_facets]
                all_annotations = self._bank_annotations
            else:
                logger.info("Generating %d facet(s) for one_shot...", max_facets)
                videos = self._knowledge_base.by_video()
                all_facets = self._extractor.init_facets(
                    task=task.name,
                    videos=videos,
                    num_facets=max_facets,
                )
                self._discovery_calls += 1

                if not all_facets:
                    logger.info("No facets generated.")
                    return

                # IRR check
                if self.config.irr_sample_videos > 0:
                    irr = compute_irr(
                        task=task.name,
                        facets=all_facets,
                        knowledge_base=self._knowledge_base,
                        labeler=self._labeler,
                        sample_videos=self.config.irr_sample_videos,
                        num_raters=self.config.irr_num_raters,
                        min_agreement=self.config.irr_min_agreement,
                    )
                    low_irr_ids = set(irr.get("low_irr_facet_ids", []))
                    if low_irr_ids:
                        all_facets = [f for f in all_facets if f.id not in low_irr_ids]
                        logger.info("Pruned %d low-IRR facet(s).", len(low_irr_ids))

                # Build vocabulary and label
                all_facets = self._vocab_builder.build_vocabulary(
                    task=task.name,
                    facets=all_facets,
                    knowledge_base=self._knowledge_base,
                    sample_videos=self.config.vocab_sample_videos,
                )
                all_annotations = self._labeler.label_batch(
                    task=task.name,
                    facets=all_facets,
                    knowledge_base=self._knowledge_base,
                )

                # Save to facet bank
                self._add_to_bank(all_facets, all_annotations)
                self._save_facet_bank(cache_dir)

            # Start pool with first target_facets, use rest as candidates
            target = self.config.target_facets
            initial_pool = all_facets[:target]
            candidates = all_facets[target:]

            for f in initial_pool:
                self._facets[f.id] = f
            self._annotations = all_annotations.filter_facets(list(self._facets.keys()))

            # Build initial tree
            self._tree = self._build_tree(focused_kb, list(self._facets.values()), self._annotations)
            logger.info(
                "Initial pool: %d facet(s), tree: %d nodes. Trying %d candidate(s)...",
                len(initial_pool), len(self._tree.nodes), len(candidates),
            )

            # Greedy selection with remaining candidates
            self._greedy_select_pool(candidates, focused_kb, all_annotations)

        logger.info(
            "one_shot complete: %d facet(s), tree: %d nodes.",
            len(self._facets), len(self._tree.nodes) if self._tree else 0,
        )

    def _run_refinement_tree(self, task: Task, focused_kb: KnowledgeBase, cache_dir: str | Path):
        """
        refinement_tree: propose facets iteratively, maintain best pool via greedy replacement.
        Stops when budget (max_facets total proposed) exhausted or max_iterations non-improvements.

        The facet bank tracks ALL discovered facets (accepted or not) so the budget
        and deduplication persist across runs with different selection parameters.
        """
        max_facets = self.config.max_facets
        max_iterations = self.config.max_iterations
        max_cells_per_iteration = self.config.max_cells_per_iteration
        max_items_per_cell = self.config.max_items_per_cell
        k_schemas_to_sample = self.config.k_schemas_to_sample
        vocab_sample_videos = self.config.vocab_sample_videos
        metric_mode = self.config.metric_mode

        # Budget tracks all facets ever discovered (bank), not just pool
        total_proposed = len(self._bank_facets)
        tries_with_no_improvement = 0

        # Build initial tree (may be empty if no facets yet)
        if self._facets and self._tree is None:
            self._tree = self._build_tree(focused_kb, list(self._facets.values()), self._annotations)

        # If no tree yet, create an empty one
        if self._tree is None:
            self._tree = RefinementTree(nodes=[], edges=[])

        # If bank has facets not yet tried for the pool, run greedy on them first
        bank_untried = [f for fid, f in self._bank_facets.items() if fid not in self._facets]
        if bank_untried:
            logger.info(
                "Bank has %d untried facet(s) — running greedy selection...",
                len(bank_untried),
            )
            self._greedy_select_pool(bank_untried, focused_kb, self._bank_annotations)
            self._save_cache(cache_dir)

        while total_proposed < max_facets and tries_with_no_improvement < max_iterations:
            self._iteration += 1
            logger.info("=== Iteration %d (proposed: %d/%d, no-improve: %d/%d) ===",
                        self._iteration, total_proposed, max_facets,
                        tries_with_no_improvement, max_iterations)

            # Print cell distribution
            if self._facets:
                cells = compute_cells(
                    focused_kb, list(self._facets.values()), self._annotations
                )
                size_counts = defaultdict(int)
                for _, units in cells.items():
                    size_counts[len(units)] += 1
                total_cell_sizes = sum(size_counts.values())
                logger.info("Cell size distribution (total=%d):", total_cell_sizes)
                for size, count in sorted(size_counts.items()):
                    logger.info(f"  {size} units: {count} ({count/total_cell_sizes*100:.1f}%)")

            # Step 1: Sample schemas from tree leaves
            sampled = sample_schemas_from_tree(
                self._tree,
                self._facets,
                k=k_schemas_to_sample,
            )
            # If no tree/leaves yet, use empty schema
            if not sampled:
                sampled = [([], None)]

            # Step 2: Facet generation
            all_new_facets: List[Facet] = []
            videos = self._knowledge_base.by_video()

            for schema_facets, _point in sampled:
                # 2a: Discover new facets from collision sites
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

            # Deduplicate by id against the bank (all previously discovered facets)
            unique_new = []
            for f in all_new_facets:
                if f.id not in self._bank_facets:
                    self._bank_facets[f.id] = f  # reserve the id to avoid dupes within batch
                    unique_new.append(f)

            if not unique_new:
                logger.info("All proposed facets already in bank.")
                tries_with_no_improvement += 1
                continue

            # Count towards budget
            total_proposed = len(self._bank_facets)
            logger.info("Candidate new facets: %d (total proposed: %d/%d)",
                        len(unique_new), total_proposed, max_facets)

            # IRR check
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
                    logger.info("Pruned %d low-IRR candidate(s).", len(low_irr_ids))
                if not unique_new:
                    logger.info("All candidates pruned by IRR.")
                    tries_with_no_improvement += 1
                    continue

            # Build vocabulary and label
            unique_new = self._vocab_builder.build_vocabulary(
                task=task.name,
                facets=unique_new,
                knowledge_base=self._knowledge_base,
                sample_videos=vocab_sample_videos,
            )
            new_annotations = self._labeler.label_batch(
                task=task.name,
                facets=unique_new,
                knowledge_base=self._knowledge_base,
            )

            # Save to facet bank (all discovered, regardless of pool acceptance)
            self._add_to_bank(unique_new, new_annotations)
            self._save_facet_bank(cache_dir)

            # Greedy pool selection
            improved = self._greedy_select_pool(unique_new, focused_kb, new_annotations)

            if improved:
                tries_with_no_improvement = 0
                self._save_cache(cache_dir)
                logger.info(
                    "Pool improved: %d facet(s), tree: %d nodes.",
                    len(self._facets), len(self._tree.nodes) if self._tree else 0,
                )
            else:
                tries_with_no_improvement += 1
                logger.info("No improvement — discarding candidates.")

        logger.info(
            "refinement_tree complete: %d iteration(s), %d facet(s), tree: %d nodes, bank: %d facets.",
            self._iteration, len(self._facets),
            len(self._tree.nodes) if self._tree else 0,
            len(self._bank_facets),
        )

    # ---- caching ----

    def _save_cache(self, cache_dir: str | Path):
        result_path = self.get_result_path(cache_dir)

        result = DiscoveryResult(
            facets=list(self._facets.values()),
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
                self._tree = result.tree
                self._discovery_calls = result.discovery_calls
                self._iteration = result.iterations
                logger.info("Restored cache from %s (%d facets, %d annotations).", result_path, len(result.facets), len(result.annotations))
        else:
            logger.info("No cache found at %s.", result_path)
