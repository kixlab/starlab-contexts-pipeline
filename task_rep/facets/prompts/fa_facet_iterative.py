"""
Prompts for iterative facet proposal: one facet at a time, conditioned on
the current pool.

The prompt is assembled from a collection of context component names:
  "facets"  — include existing pool (definitions + vocabularies + label frequencies)
  "items"   — include tutorial video segments with their current pool labels
  "schemas" — include accepted/rejected factor combinations with signature frequencies
  "history" — include previously tried factors that were not kept, with outcomes
Any non-empty combination is valid (e.g. {"facets", "items"}).
"""

import logging
from typing import Any, Collection, Dict, List, Optional

from utils.llm import MessageContent
from utils import random_uid

from .fa_criteria import FACTOR_CRITERIA
from .fa_pydantic import ContextualFactorInitResult
from .formatters import (
    build_item_id_map,
    facets_to_message_content,
    annotated_videos_to_message_content,
    schemas_to_message_content,
    history_to_message_content,
)

from facets.schema import Facet, FacetLabel, AnnotationTable
from knowledge_items.schema import KnowledgeBase, KnowledgeItem

logger = logging.getLogger(__name__)

CONTEXT_COMPONENTS = frozenset({"facets", "items", "schemas", "history"})


SYSTEM_PROMPT = """You design user-facing metadata for tutorial videos about: {task}.

You propose contextual factors: questions about a learner's situation, progress, or current step that determine which tutorial segments apply to them.

Your audience is novice and intermediate learners of {task}. Factors become filter menus, timeline labels, and search options. Learners use them to find useful segments before watching, so every factor must be something a learner can answer about themselves without seeing the footage."""

_INSTRUCTIONS = """Propose exactly ONE new contextual factor that complements the existing ones. A proposal is expected on every call; return an empty candidate list only if every candidate you can think of would duplicate an existing factor, and say which candidates you considered.

""" + FACTOR_CRITERIA + """

INPUTS
"""

_SECTION_FACETS = """- Existing contextual factors:
{facets}
"""

_SECTION_ITEMS = """- Tutorial segments (with current pool labels):
{tutorials}
"""


_SECTION_SCHEMAS = """- The current design space of factor combinations. A pruned combination does not make its factors bad; a new factor should add a distinction that the kept combinations do not already carry, alone or paired with an existing factor.
{factor_combinations}
"""

_SECTION_HISTORY = """- Previously tried factors that were not kept:
Do not re-propose these factors or close paraphrases of them.
{history}
"""

_REMINDER = """REMINDER: propose exactly one new factor; every value must complete "Show me the segments for when ___"; prefer factors that apply across most tutorials.
"""


def form_facet_iterative_request(
    task: str,
    facets: List[Facet],
    components: Collection[str],
    videos: Optional[Dict[str, List[KnowledgeItem]]] = None,
    annotations: Optional[AnnotationTable] = None,
    knowledge_base: Optional[KnowledgeBase] = None,
    accepted_schemas: Optional[List[Any]] = None,
    rejected_schemas: Optional[List[Any]] = None,
    refinement_edges: Optional[List[Any]] = None,
    refinement_threshold: Optional[float] = None,
    last_submission: Optional[Any] = None,
    history: Optional[List[Any]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Form a request to propose one new facet given the configured context components."""
    if isinstance(components, str):
        raise TypeError("components must be a collection of names, not a string.")
    unknown = set(components) - CONTEXT_COMPONENTS
    if unknown:
        raise ValueError(f"Unknown context components: {sorted(unknown)}.")

    content_blocks = MessageContent.from_template(
        _INSTRUCTIONS,
        task=task,
    ).content

    if "facets" in components:
        mc = MessageContent.from_template(
            _SECTION_FACETS,
            facets=facets_to_message_content(
                facets,
                annotations=annotations,
                knowledge_base=knowledge_base,
            ),
        )
        content_blocks.extend(mc.content)

    if "items" in components and videos:
        all_items = [item for items in videos.values() for item in items]
        id_map = build_item_id_map(all_items)
        mc = MessageContent.from_template(
            _SECTION_ITEMS,
            tutorials=annotated_videos_to_message_content(
                videos, facets, annotations or AnnotationTable(), id_map
            ),
        )
        content_blocks.extend(mc.content)

    if "schemas" in components and knowledge_base is not None:
        mc = MessageContent.from_template(
            _SECTION_SCHEMAS,
            factor_combinations=schemas_to_message_content(
                accepted_schemas or [],
                rejected_schemas or [],
                facets,
                annotations or AnnotationTable(),
                knowledge_base,
                refinement_edges,
                refinement_threshold=refinement_threshold,
                last_submission=last_submission,
            ),
        )
        content_blocks.extend(mc.content)

    if "history" in components:
        mc = MessageContent.from_template(
            _SECTION_HISTORY,
            history=history_to_message_content(history or []),
        )
        content_blocks.extend(mc.content)

    if len(content_blocks) == 1:  # No context components included
        content_blocks.append(
            {"type": "text", "text": "- No additional context provided."}
        )

    content_blocks.append({"type": "text", "text": _REMINDER})

    messages = [
        {
            "role": "system",
            "content": MessageContent.from_template(SYSTEM_PROMPT, task=task),
        },
        {
            "role": "user",
            "content": MessageContent(content=content_blocks),
        },
    ]
    return {
        "messages": messages,
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
