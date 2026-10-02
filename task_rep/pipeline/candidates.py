"""
Candidate processing lifecycle.

One lifecycle for all strategies: validate + dedup, optional IRR, vocabulary,
labeling. Returns a processed candidate or a structured failure. This module
never selects the pool or manages checkpoints — that is the runner/evaluation's
job.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Union, runtime_checkable

import numpy as np

from knowledge_items import KnowledgeBase

from facets import Facet, AnnotationTable

from optimization import compute_irr

from .config import PipelineConfig
from .models import CandidateStatus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Injectable service protocols (the real VLM classes satisfy these structurally)
# ---------------------------------------------------------------------------

@runtime_checkable
class Embedder(Protocol):
    def embed(self, texts: List[str]) -> np.ndarray: ...


@runtime_checkable
class VocabularyBuilder(Protocol):
    def build_vocabulary(
        self, task: str, facets: List[Facet], knowledge_base: KnowledgeBase,
        sample_videos: int = 0,
    ) -> List[Facet]: ...


@runtime_checkable
class Labeler(Protocol):
    def label_batch(
        self, task: str, facets: List[Facet], knowledge_base: KnowledgeBase,
        temperature: Optional[float] = None,
    ) -> AnnotationTable: ...


# ---------------------------------------------------------------------------
# Lifecycle outcomes
# ---------------------------------------------------------------------------

@dataclass
class ProcessedCandidate:
    """A candidate that passed the whole lifecycle."""

    facet: Facet
    annotations: AnnotationTable
    embedding: List[float]
    nearest_id: Optional[str] = None
    nearest_similarity: Optional[float] = None


@dataclass
class CandidateFailure:
    """A candidate that fell out of the lifecycle, with the reason."""

    status: CandidateStatus
    facet: Optional[Facet] = None
    embedding: Optional[List[float]] = None
    nearest_id: Optional[str] = None
    nearest_similarity: Optional[float] = None
    duplicate_of: Optional[str] = None
    note: str = ""


CandidateOutcome = Union[ProcessedCandidate, CandidateFailure]


def dedup_text(facet: Facet) -> str:
    """Canonical text rendering used for dedup embedding.

    Guidelines are excluded (verbose, phrasing-unstable); vocabulary does not
    exist yet at dedup time.
    """
    return f"{facet.title} ({facet.scope}): {facet.definition}"


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


@dataclass
class _Survivor:
    """A candidate past dedup, carried through the batched stages."""

    index: int
    facet: Facet
    embedding: List[float]
    nearest_id: Optional[str]
    nearest_similarity: Optional[float]

    def fail(self, status: CandidateStatus, facet: Optional[Facet] = None) -> CandidateFailure:
        return CandidateFailure(
            status=status, facet=facet or self.facet, embedding=self.embedding,
            nearest_id=self.nearest_id, nearest_similarity=self.nearest_similarity,
        )


class CandidateProcessor:
    """Runs candidates through the shared lifecycle.

    Dedup is sequential in input order; IRR, vocabulary, and labeling are each
    one call over all surviving candidates, so a0 keeps the batch runners'
    concurrency. Iterative strategies use the same path with one candidate.
    """

    def __init__(
        self,
        config: PipelineConfig,
        task: str,
        knowledge_base: KnowledgeBase,
        vocab_builder: VocabularyBuilder,
        labeler: Labeler,
        embedder: Embedder,
    ):
        self.config = config
        self.task = task
        self.knowledge_base = knowledge_base
        self.vocab_builder = vocab_builder
        self.labeler = labeler
        self.embedder = embedder

    # ---- individual steps ----

    @staticmethod
    def _validate(facet: Facet) -> bool:
        return bool(facet and facet.title and facet.title.strip()
                    and facet.definition and facet.definition.strip())

    @staticmethod
    def _nearest(
        vec: np.ndarray, existing: Dict[str, List[float]],
    ) -> tuple[Optional[str], Optional[float]]:
        best_id: Optional[str] = None
        best_sim: Optional[float] = None
        for prev_id, prev_vec in existing.items():
            sim = _cosine(vec, np.asarray(prev_vec, dtype=float))
            if best_sim is None or sim > best_sim:
                best_id, best_sim = prev_id, sim
        return best_id, best_sim

    # ---- full lifecycle ----

    def process(
        self, facet: Facet, existing_embeddings: Dict[str, List[float]],
    ) -> CandidateOutcome:
        """Run one candidate through the lifecycle."""
        return self.process_batch([facet], existing_embeddings)[0]

    def process_batch(
        self, facets: List[Facet], existing_embeddings: Dict[str, List[float]],
    ) -> List[CandidateOutcome]:
        """Validate -> dedup -> IRR -> vocabulary -> labeling.

        ``existing_embeddings`` holds candidates that already entered the
        lifecycle; it is not mutated. Returns one outcome per input, in order.
        """
        outcomes: List[Optional[CandidateOutcome]] = [None] * len(facets)

        # 1. validation + sequential dedup against prior and earlier-in-batch
        #    lifecycle entrants (never against duplicates: no chaining)
        valid = []
        for i, f in enumerate(facets):
            if self._validate(f):
                valid.append((i, f))
            else:
                outcomes[i] = CandidateFailure(status=CandidateStatus.INVALID, facet=f,
                                               note="missing title or definition")
        vectors = (self.embedder.embed([dedup_text(f) for _, f in valid])
                   if valid else [])
        seen = dict(existing_embeddings)
        survivors: List[_Survivor] = []
        for (i, f), raw in zip(valid, vectors):
            vec = np.asarray(raw, dtype=float)
            nearest_id, sim = self._nearest(vec, seen)
            if sim is not None and sim >= self.config.dedup_similarity_threshold:
                outcomes[i] = CandidateFailure(
                    status=CandidateStatus.DUPLICATE, facet=f, embedding=vec.tolist(),
                    nearest_id=nearest_id, nearest_similarity=sim, duplicate_of=nearest_id,
                )
                continue
            seen[f.id] = vec.tolist()
            survivors.append(_Survivor(i, f, vec.tolist(), nearest_id, sim))

        # 2. IRR (optional)
        if survivors and self.config.irr_sample_videos > 0:
            irr = compute_irr(
                task=self.task, facets=[s.facet for s in survivors],
                knowledge_base=self.knowledge_base, labeler=self.labeler,
                sample_videos=self.config.irr_sample_videos,
                num_raters=self.config.irr_num_raters,
                min_agreement=self.config.irr_min_agreement,
            )
            low = set(irr.get("low_irr_facet_ids", []))
            for s in survivors:
                if s.facet.id in low:
                    outcomes[s.index] = s.fail(CandidateStatus.IRR_PRUNED)
            survivors = [s for s in survivors if s.facet.id not in low]

        # 3. vocabulary
        if survivors:
            built = {f.id: f for f in self.vocab_builder.build_vocabulary(
                task=self.task, facets=[s.facet for s in survivors],
                knowledge_base=self.knowledge_base,
                sample_videos=self.config.vocab_sample_videos,
            )}
            kept = []
            for s in survivors:
                facet = built.get(s.facet.id)
                if facet is None or not facet.vocabulary:
                    outcomes[s.index] = s.fail(CandidateStatus.VOCABULARY_FAILED)
                else:
                    s.facet = facet
                    kept.append(s)
            survivors = kept

        # 4. labeling
        if survivors:
            annotations = self.labeler.label_batch(
                task=self.task, facets=[s.facet for s in survivors],
                knowledge_base=self.knowledge_base,
            )
            labeled = set(annotations.facet_ids())
            for s in survivors:
                if s.facet.id not in labeled:
                    outcomes[s.index] = s.fail(CandidateStatus.LABELING_FAILED)
                else:
                    outcomes[s.index] = ProcessedCandidate(
                        facet=s.facet,
                        annotations=annotations.filter_facets([s.facet.id]),
                        embedding=s.embedding,
                        nearest_id=s.nearest_id,
                        nearest_similarity=s.nearest_similarity,
                    )

        return outcomes  # type: ignore[return-value]
