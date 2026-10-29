"""
Shared fixtures for contextual-unit tests: a tiny knowledge base with hand-designed
label partitions, and scripted fake services that record their calls.

Knowledge base: 8 items (i1..i8) in two videos (v1: i1-i4, v2: i5-i8), each item
its own unit (u1..u8). Facets are defined by how they partition the 8 items:

    A   a a a a b b b b
    A2  same partition as A   (redundant with A)
    A3  same partition as A   (redundant with A)
    B   a a b b a a b b       (independent of A)
    D   same partition as B   (redundant with B)
    C   a b a b a b a b       (independent of A and B)

With ``max_schema_facets=1`` the refinement tree is the root node plus one node
per non-redundant facet, so tree node counts are easy to verify by hand:
pool {A} -> 2 nodes, {A, A2} -> 2, {A, B} -> 3, {A, B, C} -> 4.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Set

import numpy as np

from facets import AnnotationTable, Facet, FacetLabel
from knowledge_items import ItemScopeConfig, KnowledgeItemType
from knowledge_items.schema import KnowledgeBase, KnowledgeItem, KnowledgeSource
from utils import ModelConfig

from contextual.lifecycle import dedup_text
from contextual.settings import RunSettings


ITEM_IDS = [f"i{n}" for n in range(1, 9)]

PARTITIONS: Dict[str, str] = {
    "A": "aaaabbbb",
    "A2": "aaaabbbb",
    "A3": "aaaabbbb",
    "B": "aabbaabb",
    "D": "aabbaabb",
    "C": "abababab",
}
DEFAULT_PARTITION = "abbaabba"


def make_kb() -> KnowledgeBase:
    items = []
    for n, item_id in enumerate(ITEM_IDS, start=1):
        video = "v1" if n <= 4 else "v2"
        items.append(KnowledgeItem(
            id=item_id,
            type=KnowledgeItemType.ATOMIC,
            text=f"Segment {item_id}",
            source=KnowledgeSource(video_id=video),
            unit_id=f"u{n}",
        ))
    return KnowledgeBase(items=items)


def make_facet(
    fid: str,
    title: Optional[str] = None,
    labels: Iterable[str] = ("a", "b"),
    definition: Optional[str] = None,
    scope: str = "global",
) -> Facet:
    title = fid if title is None else title
    return Facet(
        id=fid,
        title=title,
        title_plural=f"{title}s",
        scope=scope,
        definition=f"Definition of {title}" if definition is None else definition,
        justification="test",
        guidelines=[],
        vocabulary=[FacetLabel(label=lbl, definition=f"Def {lbl}") for lbl in labels],
    )


def label_book(facet_ids: Iterable[str], partition_of: Optional[Dict[str, str]] = None
               ) -> AnnotationTable:
    """Annotations for ``facet_ids`` following PARTITIONS (or ``partition_of``).

    Facets not listed anywhere get DEFAULT_PARTITION.
    """
    partition_of = partition_of or {}
    ann = AnnotationTable()
    for fid in facet_ids:
        pattern = partition_of.get(fid, PARTITIONS.get(fid, DEFAULT_PARTITION))
        for item_id, label in zip(ITEM_IDS, pattern):
            ann[(item_id, fid)] = label
    return ann


def model_defaults() -> Dict:
    model = ModelConfig(category="vlm", provider="openai", model="fake-vlm")
    emb = ModelConfig(category="embedding", provider="openai", model="fake-emb")
    return dict(
        proposal_model=model,
        vocab_model=model,
        labeling_model=model,
        embedding_model=emb,
        item_scope=ItemScopeConfig(item_type=KnowledgeItemType.ATOMIC),
        proposal_budget=10,
        pool_size=10,
        irr_sample_videos=0,
        max_schema_facets=1,
        use_cache=False,
    )


def make_settings(**overrides) -> RunSettings:
    return RunSettings(**{**model_defaults(), **overrides})


# ---------------------------------------------------------------------------
# Scripted fakes
# ---------------------------------------------------------------------------

class ScriptedVocabBuilder:
    """Passthrough vocabulary; facets in ``fail_ids`` come back without vocabulary."""

    def __init__(self, fail_ids: Optional[Set[str]] = None):
        self.fail_ids = set(fail_ids or ())
        self.calls: List[List[str]] = []

    def build_vocabulary(self, task, facets, knowledge_base, sample_videos=0, seed=None):
        self.calls.append([f.id for f in facets])
        out = []
        for f in facets:
            if f.id in self.fail_ids:
                stripped = Facet.from_dict(f.to_dict())
                stripped.vocabulary = []
                out.append(stripped)
            else:
                out.append(f)
        return out


class ScriptedLabeler:
    """Labels from PARTITIONS (or ``partition_of``); ``fail_ids`` get no labels."""

    def __init__(self, partition_of: Optional[Dict[str, str]] = None,
                 fail_ids: Optional[Set[str]] = None):
        self.partition_of = dict(partition_of or {})
        self.fail_ids = set(fail_ids or ())
        self.calls: List[List[str]] = []

    def label_batch(self, task, facets, knowledge_base, temperature=None, seed=None):
        self.calls.append([f.id for f in facets])
        ids = [f.id for f in facets if f.id not in self.fail_ids]
        kb_items = {item.id for item in knowledge_base}
        full = label_book(ids, self.partition_of)
        out = AnnotationTable()
        for (item_id, fid), label in full.items():
            if item_id in kb_items:
                out[(item_id, fid)] = label
        return out

    def label_batch_multi(self, task, facets, knowledge_base, seeds, temperature=None):
        return [self.label_batch(task, facets, knowledge_base) for _ in seeds]


class TableEmbedder:
    """Embeds by facet title.

    ``vectors`` maps a title to an explicit vector; any other title gets its own
    one-hot dimension (cosine 0 with everything else). The title -> dimension
    map is shared by all instances so embeddings stay consistent across runner
    instances (e.g. resume after a crash). Records every text.
    """

    _onehot: Dict[str, int] = {}

    def __init__(self, vectors: Optional[Dict[str, List[float]]] = None, dim: int = 256):
        self.dim = dim
        self.vectors = {k: np.asarray(v, dtype=float) for k, v in (vectors or {}).items()}
        self.texts: List[str] = []

    def _vec(self, text: str) -> np.ndarray:
        title = text.split(" (", 1)[0]
        if title in self.vectors:
            v = self.vectors[title]
            out = np.zeros(self.dim)
            out[: len(v)] = v
            return out
        if title not in self._onehot:
            # dimensions 0-7 are reserved for explicit vectors
            self._onehot[title] = 8 + len(self._onehot)
        out = np.zeros(self.dim)
        out[self._onehot[title]] = 1.0
        return out

    def embed(self, texts: List[str]) -> np.ndarray:
        self.texts.extend(texts)
        return np.array([self._vec(t) for t in texts])


def embedding_of(embedder: TableEmbedder, facet: Facet) -> List[float]:
    return embedder.embed([dedup_text(facet)])[0].tolist()
