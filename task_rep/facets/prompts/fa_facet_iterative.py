"""
Prompts for iterative facet proposal: one facet at a time, conditioned on
the current pool.

The system message (role + every tutorial segment) is the same in every
condition (``fa_system.system_message``). The user message is assembled from a
collection of context components, none of which contains segment text:
  "facets"  — the pool: each factor's id, kind, question, definition,
              guidelines, values, and segment count per value
  "items"   — each segment's label under every pool factor, by segment id
  "schemas" — the design space (DAG) of factor combinations
  "history" — tried factors that are not in the pool, and why
"items" and "schemas" refer to factors by the ids "facets" defines, so they
require it.
"""

import logging
from typing import Any, Collection, Dict, List, Optional

from utils.llm import MessageContent
from utils import random_uid

from .fa_criteria import FACTOR_CRITERIA
from .fa_pydantic import ContextualFactorInitResult
from .fa_system import system_message
from .formatters import (
    facets_to_message_content,
    segment_labels_to_message_content,
    schemas_to_message_content,
    history_to_message_content,
)

from facets.schema import Facet, FacetLabel, AnnotationTable
from knowledge_items.schema import KnowledgeBase, KnowledgeItem

logger = logging.getLogger(__name__)

CONTEXT_COMPONENTS = frozenset({"facets", "items", "schemas", "history"})
NEEDS_FACETS = frozenset({"items", "schemas"})


_INSTRUCTIONS = """Propose exactly ONE new contextual factor that complements the existing ones. A proposal is expected on every call; return an empty candidate list only if every candidate you can think of would duplicate an existing factor, and say which candidates you considered.

""" + FACTOR_CRITERIA + """

INPUTS
"""

SECTION_FACETS = """- Contextual factors in the pool:
{facets}
"""

SECTION_ITEMS = """- Current labels of the tutorial segments (segment ids as in the tutorials; factor ids as in the pool):
{labels}
"""

SECTION_SCHEMAS = """- The current design space of factor combinations. A pruned combination does not make its factors bad; a new factor should add a distinction that the kept combinations do not already carry, alone or paired with an existing factor.
{factor_combinations}
"""

SECTION_HISTORY = """- Previously tried factors that are not in the pool:
Do not re-propose these factors or close paraphrases of them.
{history}
"""

_REMINDER = """REMINDER: propose exactly one new factor; every value must complete "Show me the segments for when ___"; prefer factors that apply across most tutorials.
"""


def facets_section(facets, annotations, knowledge_base, ids=None, template=SECTION_FACETS):
    return MessageContent.from_template(template, facets=facets_to_message_content(
        facets, annotations=annotations, knowledge_base=knowledge_base, ids=ids)).content


def items_section(corpus, facets, annotations, video_ids=None, ids=None):
    return MessageContent.from_template(SECTION_ITEMS, labels=segment_labels_to_message_content(
        corpus, facets, annotations, video_ids=video_ids, ids=ids)).content


def schemas_section(facets, annotations, knowledge_base, accepted_schemas, rejected_schemas,
                    refinement_edges=None, refinement_threshold=None, ids=None,
                    max_schema_facets=None):
    return MessageContent.from_template(
        SECTION_SCHEMAS, factor_combinations=schemas_to_message_content(
            accepted_schemas or [], rejected_schemas or [], facets,
            annotations or AnnotationTable(), knowledge_base, refinement_edges,
            refinement_threshold=refinement_threshold, ids=ids,
            max_schema_facets=max_schema_facets)).content


def history_section(history, template=SECTION_HISTORY):
    return MessageContent.from_template(
        template, history=history_to_message_content(history or [])).content


def context_sections(
    components: Collection[str],
    facets: List[Facet],
    corpus: Optional[Dict[str, List[KnowledgeItem]]] = None,
    item_video_ids: Optional[List[str]] = None,
    annotations: Optional[AnnotationTable] = None,
    knowledge_base: Optional[KnowledgeBase] = None,
    accepted_schemas: Optional[List[Any]] = None,
    rejected_schemas: Optional[List[Any]] = None,
    refinement_edges: Optional[List[Any]] = None,
    refinement_threshold: Optional[float] = None,
    history: Optional[List[Any]] = None,
    ids: Optional[Dict[str, str]] = None,
    max_schema_facets: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Content blocks for the requested context components, in a fixed order.

    ``ids`` maps facet ids to stable display ids (default: list position);
    ``item_video_ids`` restricts ITEMS to some videos (default: all).
    """
    annotations = annotations if annotations is not None else AnnotationTable()
    blocks: List[Dict[str, Any]] = []
    if "facets" in components:
        blocks += facets_section(facets, annotations, knowledge_base, ids)
    if "items" in components and corpus:
        blocks += items_section(corpus, facets, annotations, item_video_ids, ids)
    if "schemas" in components and knowledge_base is not None:
        blocks += schemas_section(facets, annotations, knowledge_base, accepted_schemas,
                                  rejected_schemas, refinement_edges, refinement_threshold, ids,
                                  max_schema_facets)
    if "history" in components:
        blocks += history_section(history)
    return blocks


def form_facet_iterative_request(
    task: str,
    facets: List[Facet],
    components: Collection[str],
    corpus: Dict[str, List[KnowledgeItem]],
    item_video_ids: Optional[List[str]] = None,
    annotations: Optional[AnnotationTable] = None,
    knowledge_base: Optional[KnowledgeBase] = None,
    accepted_schemas: Optional[List[Any]] = None,
    rejected_schemas: Optional[List[Any]] = None,
    refinement_edges: Optional[List[Any]] = None,
    refinement_threshold: Optional[float] = None,
    history: Optional[List[Any]] = None,
    ids: Optional[Dict[str, str]] = None,
    max_schema_facets: Optional[int] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Form a request to propose one new facet given the configured context components."""
    if isinstance(components, str):
        raise TypeError("components must be a collection of names, not a string.")
    unknown = set(components) - CONTEXT_COMPONENTS
    if unknown:
        raise ValueError(f"Unknown context components: {sorted(unknown)}.")
    if set(components) & NEEDS_FACETS and "facets" not in components:
        raise ValueError("The items and schemas components require facets.")

    content_blocks = MessageContent.from_template(_INSTRUCTIONS, task=task).content
    content_blocks += context_sections(
        components, facets, corpus=corpus, item_video_ids=item_video_ids,
        annotations=annotations, knowledge_base=knowledge_base,
        accepted_schemas=accepted_schemas, rejected_schemas=rejected_schemas,
        refinement_edges=refinement_edges, refinement_threshold=refinement_threshold,
        history=history, ids=ids, max_schema_facets=max_schema_facets,
    )
    if len(content_blocks) == 1:  # No context components included
        content_blocks.append({"type": "text", "text": "- No additional context provided."})
    content_blocks.append({"type": "text", "text": _REMINDER})

    return {
        "messages": [
            {"role": "system", "content": system_message(task, corpus)},
            {"role": "user", "content": MessageContent(content=content_blocks)},
        ],
        "response_format": ContextualFactorInitResult,
    }


def process_facet_iterative_response(
    response: Dict[str, Any],
    **kwargs,
) -> List[Optional[Facet]]:
    """Process iterative facet response.

    Returns a list with one element: the proposed Facet, or empty list if refused.
    batch_run_llm_calls expects a list return; the caller unpacks the first element.
    """
    result = response.get("response", {})

    refusal_reason = result.get("refusal_reason", "")
    candidates = result.get("candidates", [])

    if not candidates:
        logger.info("Iterative proposal refused: %s", refusal_reason or "(no reason given)")
        return []
    
    logger.info("Iterative proposal proposed %d facet(s).", len(candidates))

    facets = []
    for candidate in candidates:
        vocabulary = [
            FacetLabel(label=ld["label"].strip().lower(), definition=ld["definition"])
            for ld in candidate.get("vocabulary", [])
        ]
        facets.append(Facet(
            id=random_uid(),
            title=candidate["title"],
            title_plural=candidate["title_plural"],
            user_question=candidate.get("user_question", ""),
            kind=candidate.get("kind", ""),
            scope=candidate["scope"],
            definition=candidate["definition"],
            justification=candidate["justification"],
            guidelines=candidate["guidelines"],
            vocabulary=vocabulary,
        ))
    return facets
