"""
Prompt fingerprint for pipeline checkpoint identity.

Renders the proposal request the configured strategy sends — batch init, or
iterative proposal for the configured context components — together with the
shared lifecycle requests on ``contextual.fingerprint.fixture()``, and hashes
them.
"""

from __future__ import annotations

from contextual.fingerprint import (
    FIXTURE_TASK, fixture, hash_requests, lifecycle_requests,
)
from facets.prompts.fa_facet_init import form_facet_init_request
from facets.prompts.fa_facet_iterative import form_facet_iterative_request

from .config import PipelineConfig, ProposalStrategy
from .proposal import HistoryEntry


def prompt_fingerprint(config: PipelineConfig) -> str:
    """Stable hash of every prompt this configuration's run can send."""
    fx = fixture()
    kb = fx["kb"]
    videos = kb.by_video()
    if config.strategy == ProposalStrategy.BATCH:
        proposal = form_facet_init_request(task=FIXTURE_TASK, videos=videos, num_facets=3)
    else:
        history = [HistoryEntry(title="Nori Brand", scope="fine-grained",
                                definition="Brand of nori used.", outcome="duplicate",
                                duplicate_of="Rice Type")]
        proposal = form_facet_iterative_request(
            task=FIXTURE_TASK,
            facets=fx["facets"],
            components={c.value for c in config.context},
            videos=videos,
            annotations=fx["annotations"],
            knowledge_base=kb,
            accepted_schemas=fx["accepted"],
            rejected_schemas=fx["rejected"],
            history=history,
        )
    return hash_requests([proposal] + lifecycle_requests(fx))
