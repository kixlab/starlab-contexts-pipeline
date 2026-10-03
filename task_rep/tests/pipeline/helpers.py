"""
Pipeline-specific test helpers on top of ``tests.contextual.helpers``:
``make_config`` (PipelineConfig) and the scripted proposer.
"""

from __future__ import annotations

from typing import List, Optional

from facets import Facet

from pipeline.config import A1_CONTEXT, PipelineConfig, ProposalStrategy
from pipeline.proposal import BatchContext, IterativeContext

from tests.contextual.helpers import (  # noqa: F401  (re-exported for pipeline tests)
    DEFAULT_PARTITION, ITEM_IDS, PARTITIONS, ScriptedLabeler, ScriptedVocabBuilder,
    TableEmbedder, embedding_of, label_book, make_facet, make_kb, model_defaults,
)


def make_config(**overrides) -> PipelineConfig:
    defaults = dict(model_defaults(), strategy=ProposalStrategy.ITERATIVE, context=A1_CONTEXT)
    defaults.update(overrides)
    return PipelineConfig(**defaults)


class ScriptedProposer:
    """Returns scripted responses in order and records every context.

    Iterative: each script entry is a list ([] = refusal, [facet] = proposal) or
    an Exception instance to raise. Batch: ``batch`` is returned as-is.
    Once the script is exhausted, further calls refuse.
    """

    def __init__(self, script: Optional[List] = None, batch: Optional[List[Facet]] = None):
        self.script = list(script or [])
        self.batch = list(batch or [])
        self.iterative_contexts: List[IterativeContext] = []
        self.batch_contexts: List[BatchContext] = []

    def propose_batch(self, context: BatchContext) -> List[Facet]:
        self.batch_contexts.append(context)
        return list(self.batch)

    def propose_one(self, context: IterativeContext) -> List[Facet]:
        self.iterative_contexts.append(context)
        if not self.script:
            return []
        entry = self.script.pop(0)
        if isinstance(entry, Exception):
            raise entry
        return list(entry)


