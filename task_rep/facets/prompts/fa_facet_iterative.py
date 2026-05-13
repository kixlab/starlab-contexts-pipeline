"""
Prompts for iterative facet proposal: one facet at a time, conditioned on
the current pool.

Context components are enabled by substring check on the `context` string:
  "facets" — include existing pool (definitions + vocabularies)
  "items"  — include tutorial video segments with their current pool labels
  "tree"   — include refinement-tree statistics
Any combination is valid (e.g. "facets+items+tree").
"""

import logging
from typing import Any, Dict, List, Optional

from utils.llm import MessageContent
from utils import random_uid

from .fa_pydantic import ContextualFactorInitResult
from .formatters import (
    build_item_id_map,
    facets_to_message_content,
    annotated_videos_to_message_content,
    tree_stats_to_message_content,
)

from facets.schema import Facet, FacetLabel, AnnotationTable
from knowledge_items.schema import KnowledgeItem

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a helpful assistant who can analyze and detect the contextual factors around the task: {task}."""

_INSTRUCTIONS = """Your goal is to propose exactly ONE new contextual factor that complements the existing ones.
A contextual factor is an aspect of the procedural situation that helps characterize when a tutorial segment is applicable.
The new factor must:
- Be distinct from all existing factors
- Capture a meaningful dimension of variation not yet covered
- Be interpretable and broadly useful across tutorial segments
- Not be based on narration, editing, presentation style, or simple restatements of segment content
If no useful new factor can be identified, return an empty list and briefly explain why.

For each proposed contextual factor, provide:
- `title`: short name (1-3 words)
- `title_plural`: plural form of the title
- `scope`: one of "global", "mid-level", or "fine-grained"
- `definition`: brief description of what this factor captures
- `justification`: (1) why this is a valid contextual factor, and (2) why it is distinct from the previous contextual factors (if any)
- `guidelines`: list of concise instructions for labeling tutorial segments under this factor, including when the label should change
- `vocabulary`: finite list of mutually exclusive label values, each with `label` and `definition`

INPUTS
"""

_SECTION_FACETS = """- Existing contextual factors:
{facets}
"""

_SECTION_ITEMS = """- Tutorial segments (with current pool labels):
{tutorials}
"""

_SECTION_TREE = """- Refinement tree statistics:
{tree_stats}
"""


def form_facet_iterative_request(
    task: str,
    facets: List[Facet],
    context: str,
    videos: Optional[Dict[str, List[KnowledgeItem]]] = None,
    annotations: Optional[AnnotationTable] = None,
    tree_stats: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Form a request to propose one new facet given the configured context components."""
    content_blocks = [{"type": "text", "text": _INSTRUCTIONS}]

    if "facets" in context:
        mc = MessageContent.from_template(
            _SECTION_FACETS,
            facets=facets_to_message_content(facets),
        )
        content_blocks.extend(mc.content)

    if "items" in context and videos:
        all_items = [item for items in videos.values() for item in items]
        id_map = build_item_id_map(all_items)
        mc = MessageContent.from_template(
            _SECTION_ITEMS,
            tutorials=annotated_videos_to_message_content(
                videos, facets, annotations or AnnotationTable(), id_map
            ),
        )
        content_blocks.extend(mc.content)

    if "tree" in context and tree_stats is not None:
        mc = MessageContent.from_template(
            _SECTION_TREE,
            tree_stats=tree_stats_to_message_content(tree_stats),
        )
        content_blocks.extend(mc.content)

    if len(content_blocks) == 1:  # No context components included
        content_blocks.append(
            {"type": "text", "text": "- No additional context provided."}
        )

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
            scope=candidate["scope"],
            definition=candidate["definition"],
            justification=candidate["justification"],
            guidelines=candidate["guidelines"],
            vocabulary=vocabulary,
        ))
    return facets
