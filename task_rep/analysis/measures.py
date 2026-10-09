"""
Measures over a finished run that need no model calls.

Everything works on a pool of facets, their annotations, and the focused
knowledge base the run was scored on. Entropies use the tree's unit-weighted
signature entropy (``optimization.tree.compute_signature_entropy``), so they
agree with the pruning numbers.

- ``levels``: kept / pruned single factors and pairs at one threshold
  (K=1 and K=2 reported separately).
- ``partition_diversity``: threshold-free diversity of the pool's partitions:
  Vendi score with an NMI similarity kernel, mean NMI, mean normalized
  variation of information.
- ``shuffled_null``: the same measures after shuffling each factor's labels
  across segments (value distribution kept): where the metrics top out when
  factors carry no shared structure.
- ``junk_control``: the pool plus deliberately meaningless but consistent
  factors: whether the metrics reward junk.
- ``unit_coverage`` / ``applicable_entropy``: coverage weighted by knowledge
  unit, and granularity measured only where a factor applies.
- ``embedding_dispersion``: spread of the definition embeddings.
- ``saturation_curve``: levels and diversity after every pool change.
"""

from __future__ import annotations

import dataclasses
import math
import random
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from contextual.evaluation import PoolEvaluator
from contextual.settings import RunSettings
from facets import AnnotationTable, Facet, FacetLabel
from knowledge_items import KnowledgeBase
from optimization.metrics import compute_cells
from optimization.tree import compute_signature_entropy

# Labels meaning "this factor does not apply here" (or is not revealed).
NOT_APPLICABLE = re.compile(r"^(not[- ]applicable|n/?a|unspecified|none|nothing)$", re.I)


def is_applicable(label: Optional[str]) -> bool:
    return bool(label) and not NOT_APPLICABLE.match(label.strip())


def _label(ann: AnnotationTable, item_id: str, facet_id: str) -> Optional[str]:
    return ann[(item_id, facet_id)] if (item_id, facet_id) in ann else None


def _scored_items(kb: KnowledgeBase):
    return [item for item in kb if item.unit_id is not None]


# ---------------------------------------------------------------------------
# Entropy, NMI, VI
# ---------------------------------------------------------------------------

def entropy(kb: KnowledgeBase, facets: Sequence[Facet], ann: AnnotationTable) -> float:
    """Unit-weighted signature entropy of the partition by ``facets``."""
    ids = [f.id for f in facets]
    return compute_signature_entropy(compute_cells(kb, list(facets), ann.filter_facets(ids)))


def information_matrices(kb: KnowledgeBase, facets: Sequence[Facet], ann: AnnotationTable):
    """Pairwise NMI (MI over mean entropy) and normalized VI (VI over joint
    entropy) between the pool's partitions. Diagonals are 1 and 0."""
    n = len(facets)
    h = [entropy(kb, [f], ann) for f in facets]
    nmi, nvi = np.eye(n), np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            joint = entropy(kb, [facets[i], facets[j]], ann)
            mi = max(0.0, h[i] + h[j] - joint)
            mean_h = (h[i] + h[j]) / 2
            nmi[i, j] = nmi[j, i] = mi / mean_h if mean_h > 0 else 1.0
            nvi[i, j] = nvi[j, i] = (joint - mi) / joint if joint > 0 else 0.0
    return nmi, nvi, h


def vendi(similarity: np.ndarray) -> float:
    """Vendi score: exp of the entropy of the eigenvalues of K/n.

    1 when all items are identical, n when they are all unrelated. NMI
    kernels are not guaranteed positive semidefinite, so negative eigenvalues
    are clipped and the rest renormalized.
    """
    n = similarity.shape[0]
    if n == 0:
        return 0.0
    eig = np.clip(np.linalg.eigvalsh(similarity / n), 0.0, None)
    total = eig.sum()
    if total <= 0:
        return 0.0
    p = eig[eig > 0] / total
    return float(math.exp(-(p * np.log(p)).sum()))


def _off_diagonal_mean(m: np.ndarray) -> float:
    n = m.shape[0]
    if n < 2:
        return 0.0
    return float((m.sum() - np.trace(m)) / (n * (n - 1)))


def partition_diversity(kb: KnowledgeBase, facets: Sequence[Facet],
                        ann: AnnotationTable) -> Dict[str, float]:
    nmi, nvi, _ = information_matrices(kb, facets, ann)
    return {
        "vendi_nmi": vendi(nmi),
        "mean_nmi": _off_diagonal_mean(nmi),
        "mean_nvi": _off_diagonal_mean(nvi),
    }


# ---------------------------------------------------------------------------
# Tree levels (K=1 / K=2)
# ---------------------------------------------------------------------------

def levels(kb: KnowledgeBase, facets: Sequence[Facet], ann: AnnotationTable,
           settings: RunSettings, tau: Optional[float] = None) -> Dict[str, Any]:
    """Kept and pruned single factors and pairs at ``tau`` (default: the run's)."""
    config = settings if tau is None else dataclasses.replace(settings, refinement_threshold=tau)
    if not facets:
        return {"kept_singles": 0, "kept_pairs": 0, "pruned_singles": 0, "pruned_pairs": 0,
                "kept_single_ids": [], "kept_pair_ids": []}
    evaluation = PoolEvaluator(config=config, focused_kb=kb).evaluate_pool(list(facets), ann)
    kept = [s for s in evaluation.accepted_schemas if s.facet_ids]
    pruned = evaluation.rejected_schemas
    return {
        "kept_singles": sum(len(s.facet_ids) == 1 for s in kept),
        "kept_pairs": sum(len(s.facet_ids) == 2 for s in kept),
        "pruned_singles": sum(len(s.facet_ids) == 1 for s in pruned),
        "pruned_pairs": sum(len(s.facet_ids) == 2 for s in pruned),
        "kept_single_ids": [s.facet_ids[0] for s in kept if len(s.facet_ids) == 1],
        "kept_pair_ids": [list(s.facet_ids) for s in kept if len(s.facet_ids) == 2],
    }


def summary(kb, facets, ann, settings) -> Dict[str, Any]:
    out = {k: v for k, v in levels(kb, facets, ann, settings).items() if not k.endswith("_ids")}
    out.update(partition_diversity(kb, facets, ann))
    return out


# ---------------------------------------------------------------------------
# Null and junk controls
# ---------------------------------------------------------------------------

def shuffle_labels(kb: KnowledgeBase, facets: Sequence[Facet], ann: AnnotationTable,
                   rng: random.Random) -> AnnotationTable:
    """Each factor's labels permuted across the scored segments it labels."""
    items = [item.id for item in _scored_items(kb)]
    shuffled = AnnotationTable()
    for facet in facets:
        labeled = [i for i in items if (i, facet.id) in ann]
        labels = [ann[(i, facet.id)] for i in labeled]
        rng.shuffle(labels)
        for item_id, label in zip(labeled, labels):
            shuffled[(item_id, facet.id)] = label
    return shuffled


def _mean_sd(rows: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    out = {}
    for key in rows[0]:
        values = [r[key] for r in rows]
        mean = sum(values) / len(values)
        sd = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
        out[key] = {"mean": mean, "sd": sd}
    return out


def shuffled_null(kb, facets, ann, settings, repeats: int = 20,
                  seed: int = 0) -> Dict[str, Dict[str, float]]:
    rng = random.Random(seed)
    rows = [summary(kb, facets, shuffle_labels(kb, facets, ann, rng), settings)
            for _ in range(repeats)]
    return _mean_sd(rows)


def junk_facets(kb: KnowledgeBase, seed: int = 0):
    """Three meaningless but consistent factors and their labels: two drawn
    at random per video (2 and 3 values), one at random per segment (4)."""
    rng = random.Random(seed)
    specs = [("junk_video_2", "video", 2), ("junk_video_3", "video", 3),
             ("junk_segment_4", "segment", 4)]
    facets, ann = [], AnnotationTable()
    videos = sorted({item.source.video_id for item in kb})
    for fid, level, k in specs:
        values = [f"value {c}" for c in "abcd"[:k]]
        facets.append(Facet(
            id=fid, title=fid, title_plural=fid, scope="global", definition="junk",
            justification="junk control", guidelines=[],
            vocabulary=[FacetLabel(label=v, definition=v) for v in values]))
        per_video = {vid: rng.choice(values) for vid in videos}
        for item in kb:
            ann[(item.id, fid)] = (per_video[item.source.video_id] if level == "video"
                                   else rng.choice(values))
    return facets, ann


def junk_control(kb, facets, ann, settings, seed: int = 0) -> Dict[str, Any]:
    """Pool metrics with and without the junk factors."""
    junk, junk_ann = junk_facets(kb, seed)
    with_junk = list(facets) + junk
    merged = ann.merge(junk_ann)
    lv = levels(kb, with_junk, merged, settings)
    junk_ids = {f.id for f in junk}
    out = {k: v for k, v in lv.items() if not k.endswith("_ids")}
    out.update(partition_diversity(kb, with_junk, merged))
    out["junk_kept_singles"] = sum(fid in junk_ids for fid in lv["kept_single_ids"])
    out["junk_kept_pairs"] = sum(any(fid in junk_ids for fid in pair)
                                 for pair in lv["kept_pair_ids"])
    return out


# ---------------------------------------------------------------------------
# Coverage and granularity
# ---------------------------------------------------------------------------

def unit_coverage(kb: KnowledgeBase, facet: Facet, ann: AnnotationTable) -> float:
    """Mean over knowledge units of the share of their segments that get a
    real value (not "not applicable" / "unspecified")."""
    per_unit = defaultdict(list)
    for item in _scored_items(kb):
        per_unit[item.unit_id].append(is_applicable(_label(ann, item.id, facet.id)))
    if not per_unit:
        return 0.0
    return sum(sum(v) / len(v) for v in per_unit.values()) / len(per_unit)


def applicable_entropy(kb: KnowledgeBase, facet: Facet, ann: AnnotationTable) -> float:
    """Entropy of the factor over only the segments where it applies."""
    applicable = [item for item in _scored_items(kb)
                  if is_applicable(_label(ann, item.id, facet.id))]
    if not applicable:
        return 0.0
    return entropy(KnowledgeBase(items=applicable), [facet], ann)


def _spread(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "sd": 0.0, "min": 0.0, "max": 0.0}
    mean = sum(values) / len(values)
    return {"mean": mean, "sd": math.sqrt(sum((v - mean) ** 2 for v in values) / len(values)),
            "min": min(values), "max": max(values)}


def granularity(kb, facets, ann) -> Dict[str, Any]:
    return {
        "applicable_entropy": _spread([applicable_entropy(kb, f, ann) for f in facets]),
        "vocabulary_size": _spread([float(sum(is_applicable(v.label) for v in f.vocabulary))
                                    for f in facets]),
    }


# ---------------------------------------------------------------------------
# Semantic dispersion
# ---------------------------------------------------------------------------

def embedding_dispersion(vectors: Sequence[Sequence[float]]) -> Dict[str, float]:
    """Mean pairwise cosine distance and cosine-kernel Vendi score of the
    definition embeddings (``"{title} ({scope}): {definition}"``)."""
    if len(vectors) < 2:
        return {"mean_cosine_distance": 0.0, "vendi_cosine": float(len(vectors))}
    m = np.asarray(vectors, dtype=float)
    m = m / np.linalg.norm(m, axis=1, keepdims=True)
    cos = np.clip(m @ m.T, -1.0, 1.0)
    return {"mean_cosine_distance": 1.0 - _off_diagonal_mean(cos),
            "vendi_cosine": vendi(np.clip(cos, 0.0, None))}


# ---------------------------------------------------------------------------
# Saturation
# ---------------------------------------------------------------------------

def saturation_curve(kb, pool: Sequence[Facet], ann: AnnotationTable, ledger,
                     settings: RunSettings) -> List[Dict[str, Any]]:
    """Levels and diversity after each ledger record (the pool at that time).

    ``pool`` must hold every facet that was ever in the pool; with
    ``pool_size >= proposal_budget`` that is the final pool.
    """
    by_id = {f.id: f for f in pool}
    rows, accepted = [], 0
    cache: Dict[tuple, Dict[str, Any]] = {}
    for record in ledger:
        snap = record.tree_quality
        if snap is None:
            continue
        if record.status.value in ("accepted", "replaced"):
            accepted += 1
        ids = tuple(fid for fid in snap.pool_ids if fid in by_id)
        if ids not in cache:
            facets = [by_id[fid] for fid in ids]
            cache[ids] = summary(kb, facets, ann.filter_facets(list(ids)), settings)
        rows.append({"attempt": record.attempt_index, "accepted": accepted,
                     "pool": len(ids), **cache[ids]})
    return rows


def run_measures(kb: KnowledgeBase, pool: Sequence[Facet], ann: AnnotationTable, ledger,
                 settings: RunSettings, embeddings: Optional[Dict[str, Sequence[float]]] = None,
                 taus: Sequence[float] = (0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7),
                 null_repeats: int = 20, seed: int = 0) -> Dict[str, Any]:
    """Every measure above for one finished run."""
    pool = list(pool)
    vectors = [embeddings[f.id] for f in pool if embeddings and f.id in embeddings]
    coverage = [unit_coverage(kb, f, ann) for f in pool]
    return {
        "at_run_tau": summary(kb, pool, ann, settings),
        "tau_curve": [{"tau": t, **{k: v for k, v in levels(kb, pool, ann, settings, t).items()
                                    if not k.endswith("_ids")}} for t in taus],
        "shuffled_null": shuffled_null(kb, pool, ann, settings, null_repeats, seed),
        "junk_control": junk_control(kb, pool, ann, settings, seed),
        "unit_coverage": {"per_factor": {f.id: c for f, c in zip(pool, coverage)},
                          "mean": sum(coverage) / len(coverage) if coverage else 0.0},
        "granularity": granularity(kb, pool, ann),
        "embedding_dispersion": embedding_dispersion(vectors),
        "embeddings_found": len(vectors),
        "kinds": dict(Counter(getattr(f, "kind", "") or "unknown" for f in pool)),
        "saturation": saturation_curve(kb, pool, ann, ledger, settings),
    }
