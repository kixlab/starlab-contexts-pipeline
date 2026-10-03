"""
Inter-Rater Reliability (IRR) diagnostic for facet labeling.

Measures labeling consistency by re-labeling a sample of tutorials several
times — each pass with its own sampling seed at the model's default
temperature — and computing agreement across the independent passes.

Facets with low agreement are pruned by the pipeline's candidate lifecycle.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from typing import Any, Dict, List, Set

from knowledge_items import KnowledgeBase
from facets import Facet, AnnotationTable
from facets.label import VLMFacetLabeler

logger = logging.getLogger(__name__)

# Temperature for IRR re-labeling passes: None = the provider's default.
# A fixed low temperature (formerly 0.2) made Gemini 3 thinking run away
# (~63k thinking tokens, ~190 s per call), and with the shared request seed the
# passes were identical, so kappa was always ~1.
IRR_TEMPERATURE = None

# Rater i labels with seed IRR_BASE_SEED + i, so passes are independent of each
# other (and of the main labeling pass) yet reproducible.
IRR_BASE_SEED = 1000


def _fleiss_kappa(
    ratings: List[List[str]],
    categories: Set[str],
) -> float:
    """
    Compute Fleiss' kappa for inter-rater agreement.

    Args:
        ratings: List of items, where each item is a list of N rater labels.
                 All items must have the same number of raters.
        categories: Set of all possible label values.

    Returns:
        Fleiss' kappa in [-1, 1].  1 = perfect agreement, 0 = chance,
        negative = worse than chance.
    """
    if not ratings or not categories:
        return 0.0

    n_items = len(ratings)
    n_raters = len(ratings[0])
    if n_raters < 2 or n_items == 0:
        return 0.0

    cat_list = sorted(categories)
    cat_idx = {c: i for i, c in enumerate(cat_list)}
    n_cats = len(cat_list)

    # Build count matrix: n_items x n_cats
    counts = [[0] * n_cats for _ in range(n_items)]
    for i, item_ratings in enumerate(ratings):
        for label in item_ratings:
            if label in cat_idx:
                counts[i][cat_idx[label]] += 1

    # P_i = proportion of agreeing pairs for item i
    p_items = []
    for i in range(n_items):
        s = sum(c * c for c in counts[i]) - n_raters
        p_items.append(s / (n_raters * (n_raters - 1)) if n_raters > 1 else 0)

    p_bar = sum(p_items) / n_items if n_items > 0 else 0

    # p_j = proportion of all assignments to category j
    total_assignments = n_items * n_raters
    p_cats = []
    for j in range(n_cats):
        col_sum = sum(counts[i][j] for i in range(n_items))
        p_cats.append(col_sum / total_assignments if total_assignments > 0 else 0)

    p_e = sum(p ** 2 for p in p_cats)

    if abs(1 - p_e) < 1e-10:
        return 1.0  # Perfect agreement or degenerate case

    return (p_bar - p_e) / (1 - p_e)


def compute_irr(
    task: str,
    facets: List[Facet],
    knowledge_base: KnowledgeBase,
    labeler: VLMFacetLabeler,
    sample_videos: int = 3,
    num_raters: int = 3,
    min_agreement: float = 0.6,
) -> Dict[str, Any]:
    """
    Compute inter-rater reliability for facet labeling.

    Samples a subset of tutorials, re-labels them ``num_raters`` times (one
    seed per rater, in a single batch), and computes Fleiss' kappa per facet.

    Args:
        task: Task name.
        facets: Facets to evaluate.
        knowledge_base: Full knowledge base.
        labeler: VLMFacetLabeler instance.
        sample_videos: Number of tutorials to sample.
        num_raters: Number of independent labeling passes.
        min_agreement: Kappa threshold below which a facet is flagged.

    Returns:
        Dict with keys:
          - per_facet_kappa: {facet_id: float}
          - per_facet_agreement: {facet_id: float}  (% exact agreement)
          - overall_kappa: float
          - overall_agreement: float
          - sample_video_ids: List[str]
          - low_irr_facet_ids: List[str]  (facets with kappa < min_agreement)
    """
    if not facets or sample_videos < 1 or num_raters < 2:
        return {
            "per_facet_kappa": {},
            "per_facet_agreement": {},
            "overall_kappa": 0.0,
            "overall_agreement": 0.0,
            "sample_video_ids": [],
            "low_irr_facet_ids": [],
        }

    # Sample videos
    all_video_ids = list(knowledge_base.by_video().keys())
    k = min(sample_videos, len(all_video_ids))
    sampled_video_ids = random.sample(all_video_ids, k)
    logger.info("IRR: sampling %d videos: %s", k, sampled_video_ids)

    # Filter KB to sampled videos
    sampled_items = [
        item for item in knowledge_base
        if item.source.video_id in set(sampled_video_ids)
    ]
    sampled_kb = KnowledgeBase(items=sampled_items)

    # Collect N independent labeling passes in one batch
    logger.info("IRR: %d labeling passes", num_raters)
    passes: List[AnnotationTable] = labeler.label_batch_multi(
        task=task,
        facets=facets,
        knowledge_base=sampled_kb,
        seeds=[IRR_BASE_SEED + i for i in range(num_raters)],
        temperature=IRR_TEMPERATURE,
    )

    # Compute per-facet agreement
    per_facet_kappa: Dict[str, float] = {}
    per_facet_agreement: Dict[str, float] = {}

    item_ids = [item.id for item in sampled_kb]

    for facet in facets:
        # Collect all labels assigned to category for this facet
        all_categories: Set[str] = set()
        ratings: List[List[str]] = []

        for item_id in item_ids:
            item_labels = []
            for anno in passes:
                key = (item_id, facet.id)
                label = anno[key] if key in anno else "na"
                item_labels.append(label)
                all_categories.add(label)
            ratings.append(item_labels)

        if not ratings:
            per_facet_kappa[facet.id] = 0.0
            per_facet_agreement[facet.id] = 0.0
            continue

        kappa = _fleiss_kappa(ratings, all_categories)
        per_facet_kappa[facet.id] = kappa

        # % exact agreement: fraction of items where all raters agree
        n_agree = sum(1 for r in ratings if len(set(r)) == 1)
        per_facet_agreement[facet.id] = n_agree / len(ratings) if ratings else 0.0

    # Overall: average kappa across facets
    kappas = list(per_facet_kappa.values())
    overall_kappa = sum(kappas) / len(kappas) if kappas else 0.0
    agreements = list(per_facet_agreement.values())
    overall_agreement = sum(agreements) / len(agreements) if agreements else 0.0

    low_irr = [fid for fid, k in per_facet_kappa.items() if k < min_agreement]

    logger.info(
        "IRR: overall kappa=%.3f, agreement=%.1f%%, low-IRR facets=%d/%d",
        overall_kappa, overall_agreement * 100, len(low_irr), len(facets),
    )
    for fid in low_irr:
        logger.info(
            "  Low IRR: facet %s  kappa=%.3f  agreement=%.1f%%",
            fid, per_facet_kappa[fid], per_facet_agreement[fid] * 100,
        )

    return {
        "per_facet_kappa": per_facet_kappa,
        "per_facet_agreement": per_facet_agreement,
        "overall_kappa": overall_kappa,
        "overall_agreement": overall_agreement,
        "sample_video_ids": sampled_video_ids,
        "low_irr_facet_ids": low_irr,
    }
