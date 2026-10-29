"""
Prompt fingerprint for checkpoint identity.

Renders every request a run can issue — batch init or iterative proposal (for
the configured context components), vocabulary proposal and consolidation, and
labeling (also used by IRR) — on a tiny fixed fixture, and hashes the rendered
messages plus response schemas. This catches template *and* formatter-logic
changes while ignoring comments and refactors that do not change prompts.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List

from facets import AnnotationTable, Facet, FacetLabel
from facets.prompts.fa_facet_init import form_facet_init_request
from facets.prompts.fa_facet_iterative import form_facet_iterative_request
from facets.prompts.fa_facet_labeling import form_labeling_request
from facets.prompts.fa_facet_vocabulary import (
    form_vocabulary_consolidation_request,
    form_vocabulary_proposal_request,
)
from knowledge_items.schema import (
    KnowledgeBase,
    KnowledgeItem,
    KnowledgeItemType,
    KnowledgeSource,
)
from optimization import AcceptedSchemaSummary, RejectedSchemaSummary
from utils.llm import MessageContent

from .config import PipelineConfig, ProposalStrategy
from .proposal import HistoryEntry

_TASK = "fingerprint task"


def _fixture() -> Dict[str, Any]:
    items = [
        KnowledgeItem(
            id=f"i{n}",
            type=KnowledgeItemType.ATOMIC,
            text=f"Fixture segment {n}",
            source=KnowledgeSource(video_id="v1" if n <= 2 else "v2"),
            unit_id=f"u{n}",
        )
        for n in range(1, 5)
    ]
    kb = KnowledgeBase(items=items)

    def facet(fid: str, title: str, labels: List[str]) -> Facet:
        return Facet(
            id=fid,
            title=title,
            title_plural=f"{title}s",
            user_question=f"Which {title.lower()} applies?",
            scope="global",
            definition=f"The {title.lower()} in the learner's situation.",
            justification="Fixture.",
            guidelines=[f"Label the {title.lower()} shown."],
            vocabulary=[FacetLabel(label=l, definition=f"{l} {title.lower()}") for l in labels],
        )

    f1 = facet("f1", "Rice Type", ["white", "brown"])
    f2 = facet("f2", "Knife", ["chef", "paring"])
    ann = AnnotationTable()
    for item, (l1, l2) in zip(items, [("white", "chef"), ("brown", "chef"),
                                      ("white", "paring"), ("brown", "paring")]):
        ann[(item.id, "f1")] = l1
        ann[(item.id, "f2")] = l2

    accepted = [AcceptedSchemaSummary(id="f1", facet_ids=["f1"], size=2, D=0.5,
                                      eta2=0.9, H=1.0)]
    rejected = [RejectedSchemaSummary(
        id="f2", facet_ids=["f2"], size=2, D=0.5, eta2=0.9, H=1.0,
        rejected_by_id="f1", rejected_by_facet_ids=["f1"], MI=0.5,
        refinement_score=0.5, refinement_threshold=0.2,
    )]
    history = [HistoryEntry(title="Nori Brand", scope="fine-grained",
                            definition="Brand of nori used.", outcome="duplicate",
                            duplicate_of="Rice Type")]
    return dict(kb=kb, facets=[f1, f2], annotations=ann, accepted=accepted,
                rejected=rejected, history=history)


def _requests(config: PipelineConfig) -> List[Dict[str, Any]]:
    fx = _fixture()
    kb: KnowledgeBase = fx["kb"]
    videos = kb.by_video()
    facet = fx["facets"][0]
    video_items = videos["v1"]

    if config.strategy == ProposalStrategy.BATCH:
        proposal = form_facet_init_request(task=_TASK, videos=videos, num_facets=3)
    else:
        proposal = form_facet_iterative_request(
            task=_TASK,
            facets=fx["facets"],
            components={c.value for c in config.context},
            videos=videos,
            annotations=fx["annotations"],
            knowledge_base=kb,
            accepted_schemas=fx["accepted"],
            rejected_schemas=fx["rejected"],
            history=fx["history"],
        )
    return [
        proposal,
        form_vocabulary_proposal_request(task=_TASK, facet=facet, video_items=video_items),
        form_vocabulary_consolidation_request(
            task=_TASK, facet=facet,
            proposals_by_video={"v1": facet.vocabulary, "v2": fx["facets"][1].vocabulary},
        ),
        form_labeling_request(task=_TASK, facet=facet, video_items=video_items),
    ]


def _canonical(request: Dict[str, Any]) -> Dict[str, Any]:
    messages = []
    for m in request["messages"]:
        content = m["content"]
        if isinstance(content, MessageContent):
            content = content.content
        messages.append({"role": m["role"], "content": content})
    schema = request.get("response_format")
    return {
        "messages": messages,
        "response_format": schema.model_json_schema() if schema is not None else None,
    }


def prompt_fingerprint(config: PipelineConfig) -> str:
    """Stable hash of every prompt this configuration's run can send."""
    payload = [_canonical(r) for r in _requests(config)]
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
