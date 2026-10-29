"""
Corpus selection: optionally restrict a knowledge base to a fixed sample of
videos (e.g. a 15-video pilot of a large task) without re-running extraction
or clustering. Units keep the cluster ids of the full corpus.

The sample depends only on ``corpus_seed``, not on the run seed, so every
condition and replicate of an experiment sees the same videos.
"""

from __future__ import annotations

import random

from knowledge_items import KnowledgeBase

from .settings import RunSettings


def select_corpus(knowledge_base: KnowledgeBase, settings: RunSettings) -> KnowledgeBase:
    n = settings.corpus_sample_videos
    videos = sorted(knowledge_base.by_video())
    if n <= 0 or n >= len(videos):
        return knowledge_base
    chosen = set(random.Random(settings.corpus_seed).sample(videos, n))
    return KnowledgeBase(items=[i for i in knowledge_base if i.source.video_id in chosen])
