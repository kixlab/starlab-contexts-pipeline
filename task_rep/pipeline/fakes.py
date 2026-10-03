"""
Deterministic fake services for offline runs and tests — no VLM/network calls.

These satisfy the same protocols as the real VLM services:
- ``ReplayProposer``          -> FacetProposer
- ``PassthroughVocabularyBuilder`` -> VocabularyBuilder
- ``ReplayLabeler``           -> Labeler
- ``HashingEmbedder``         -> Embedder

``load_corpus`` reads a cached ``run_result_*.json`` and returns its facets +
annotations, so the fakes replay real discovered facets, vocabularies, and
labels instead of calling models.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np

from knowledge_items import KnowledgeBase

from facets import Facet, AnnotationTable

from .proposal import BatchContext, IterativeContext


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def load_corpus(run_result_path: str | Path) -> tuple[List[Facet], AnnotationTable]:
    """Load facets + annotations from a cached run_result JSON."""
    data = json.loads(Path(run_result_path).read_text())
    facets = [Facet.from_dict(f) for f in data["facets"]]
    annotations = AnnotationTable.from_list(data["annotations"])
    return facets, annotations


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

class HashingEmbedder:
    """Deterministic bag-of-words hashing embedder.

    Identical text -> identical vector (cosine 1.0); shared words -> higher
    cosine. Enough to exercise dedup offline without an embedding API.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=float)
        for token in text.lower().split():
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        return vec

    def embed(self, texts: List[str]) -> np.ndarray:
        return np.array([self._embed_one(t) for t in texts])


# ---------------------------------------------------------------------------
# Vocabulary builder
# ---------------------------------------------------------------------------

class PassthroughVocabularyBuilder:
    """Returns facets unchanged (corpus facets already carry vocabulary).

    ``fail_ids`` simulates vocabulary-building failure for specific facets.
    """

    def __init__(self, fail_ids: Optional[Set[str]] = None):
        self.fail_ids = fail_ids or set()

    def build_vocabulary(
        self, task: str, facets: List[Facet], knowledge_base: KnowledgeBase,
        sample_videos: int = 0, seed: Optional[int] = None,
    ) -> List[Facet]:
        out = []
        for f in facets:
            if f.id in self.fail_ids:
                stripped = Facet.from_dict(f.to_dict())
                stripped.vocabulary = []
                out.append(stripped)
            else:
                out.append(f)
        return out


# ---------------------------------------------------------------------------
# Labeler
# ---------------------------------------------------------------------------

class ReplayLabeler:
    """Replays cached annotations for the requested facets."""

    def __init__(self, annotations: AnnotationTable):
        self._annotations = annotations

    def label_batch(
        self, task: str, facets: List[Facet], knowledge_base: KnowledgeBase,
        temperature: Optional[float] = None, seed: Optional[int] = None,
    ) -> AnnotationTable:
        return self._annotations.filter_facets([f.id for f in facets])

    def label_batch_multi(
        self, task: str, facets: List[Facet], knowledge_base: KnowledgeBase,
        seeds: List[Optional[int]], temperature: Optional[float] = None,
    ) -> List[AnnotationTable]:
        """Replay is deterministic: every pass gets the cached labels."""
        return [self.label_batch(task, facets, knowledge_base) for _ in seeds]


# ---------------------------------------------------------------------------
# Proposer
# ---------------------------------------------------------------------------

class ReplayProposer:
    """Replays corpus facets as proposals.

    Batch (a0): returns the first ``num_facets`` corpus facets.
    Iterative (a1/a2): returns one facet per call in corpus order, [] when
    exhausted. ``refuse_at`` forces a refusal ([]) at the given 0-based call
    indices without consuming a facet, to exercise refusal handling.
    """

    def __init__(self, corpus: List[Facet], refuse_at: Optional[Set[int]] = None):
        self.corpus = corpus
        self.refuse_at = refuse_at or set()
        self._cursor = 0
        self._calls = 0

    def propose_batch(self, context: BatchContext) -> List[Facet]:
        return self.corpus[: context.num_facets]

    def propose_one(self, context: IterativeContext) -> List[Facet]:
        call = self._calls
        self._calls += 1
        if call in self.refuse_at:
            return []
        if self._cursor >= len(self.corpus):
            return []
        facet = self.corpus[self._cursor]
        self._cursor += 1
        return [facet]

    def prompt_fingerprint(self) -> str:
        return "fake-replay-v1"
