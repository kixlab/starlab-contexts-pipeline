"""
Prompt fingerprinting helpers shared by all runners.

Runners render every request they can issue on the fixed ``fixture()`` — their
own proposal requests plus the shared ``lifecycle_requests()`` (vocabulary
proposal and consolidation, labeling, which IRR also uses) — and hash the
rendered messages and response schemas with ``hash_requests``. This catches
template *and* formatter-logic changes while ignoring comments and refactors
that do not change prompts.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List

from facets import AnnotationTable, Facet, FacetLabel
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
from optimization import AcceptedSchemaSummary, RefinementTree, RejectedSchemaSummary, TreeEdge
from utils.llm import MessageContent

FIXTURE_TASK = "fingerprint task"


def fixture() -> Dict[str, Any]:
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

    def facet(fid: str, title: str, labels: List[str], kind: str) -> Facet:
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
            kind=kind,
        )

    f1 = facet("f1", "Rice Type", ["white", "brown"], "situation")
    f2 = facet("f2", "Knife", ["chef", "paring"], "moment")
    ann = AnnotationTable()
    for item, (l1, l2) in zip(items, [("white", "chef"), ("brown", "chef"),
                                      ("white", "paring"), ("brown", "paring")]):
        ann[(item.id, "f1")] = l1
        ann[(item.id, "f2")] = l2

    accepted = [
        AcceptedSchemaSummary(id="f1", facet_ids=["f1"], size=2, D=0.5, eta2=0.9, H=1.0),
        AcceptedSchemaSummary(id="f1|f2", facet_ids=["f1", "f2"], size=4, D=0.0,
                              eta2=0.5, H=2.0, closest_id="f1", closest_facet_ids=["f1"],
                              closest_overlap=0.5, member_overlaps={"f1": 0.5, "f2": 0.5}),
    ]
    rejected = [RejectedSchemaSummary(
        id="f2", facet_ids=["f2"], size=2, D=0.5, eta2=0.9, H=1.0,
        rejected_by_id="f1", rejected_by_facet_ids=["f1"], MI=0.5,
        refinement_score=0.5, refinement_threshold=0.2,
    )]
    tree = RefinementTree(nodes=[], edges=[TreeEdge(
        source_id="f1", target_id="f1|f2", MI=1.0, refinement_score=1.0,
        marginal_D_gain=0.5, marginal_eta2_gain=0.4, added_facets=["f2"], removed_facets=[],
    )])
    return dict(kb=kb, facets=[f1, f2], annotations=ann, accepted=accepted,
                rejected=rejected, tree=tree,
                edges=[(e.source_id, e.target_id) for e in tree.edges],
                ids={"f1": "F1", "f2": "F3"})


def lifecycle_requests(fx: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Vocabulary and labeling requests rendered on the fixture."""
    videos = fx["kb"].by_video()
    facet = fx["facets"][0]
    video_items = videos["v1"]
    return [
        form_vocabulary_proposal_request(task=FIXTURE_TASK, facet=facet, video_items=video_items),
        form_vocabulary_consolidation_request(
            task=FIXTURE_TASK, facet=facet,
            proposals_by_video={"v1": facet.vocabulary, "v2": fx["facets"][1].vocabulary},
        ),
        form_labeling_request(task=FIXTURE_TASK, facet=facet, video_items=video_items),
    ]


def canonical(request: Dict[str, Any]) -> Dict[str, Any]:
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


def hash_requests(requests: List[Dict[str, Any]], extra: Any = None) -> str:
    """Stable hash of rendered requests (plus any JSON-serializable ``extra``)."""
    rendered = [canonical(r) for r in requests]
    # Without ``extra`` the payload is the bare list, as pipeline fingerprints
    # have always been hashed (keeps existing checkpoints valid).
    payload = rendered if extra is None else {"requests": rendered, "extra": extra}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
