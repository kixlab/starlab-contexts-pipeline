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

You propose contextual factors: dimensions of a learner's situation that determine which tutorial segments apply to them. A learner's situation consists of their goal, materials, equipment, and progress through the task.

Your audience is novice and intermediate learners of {task}. Factors become filter menus, timeline labels, and search options. Learners use them to find useful segments before watching, so every factor must be something a learner can state about their own situation without seeing the footage."""

_INSTRUCTIONS = """Propose exactly ONE new contextual factor that complements the existing ones. Always propose the single best remaining candidate, even if it is narrower or finer-grained than the existing factors; when broad dimensions are covered, subdivide the learner's situation more finely (specific materials, equipment variants, preparation states, sub-goals, problem situations). Return an empty list only if every conceivable candidate would duplicate an existing factor outright.

WHAT QUALIFIES AS A CONTEXTUAL FACTOR
Each vocabulary value must be a situation a learner could truthfully claim before watching. Test every value with this sentence:

    "Show me the segments for when ___."

- If a value can only be determined by observing how the instructor moves, holds, paces, presents, or what the camera shows, it describes the footage, not a learner's situation.
- A factor is useful only if learners in different situations need different segments. Before proposing it, identify two concrete learners whose situations differ on this factor and who would retrieve different segments.
- Values must use the words of {task}, not vocabulary from robotics, motor control, cinematography, or another discipline.
- If the factor and vocabulary would transfer unchanged to an unrelated manual task such as pottery, knitting, surgery, or furniture assembly, reject it.

VALIDITY REQUIREMENTS
- Distinct from existing factors and not largely predictable from their combination.
- Grounded in the provided segments, with at least two values that would be used.
- Not based on narration, editing, camera framing, viewer visibility, visual occlusion, or segment presentation.

EXPRESSION REQUIREMENTS
- `title`: 1-3 everyday words; a concrete noun phrase a learner would recognize.
- `user_question`: the learner-facing question this factor answers, e.g. "What kind of roll are you making?" or "What tool do you have?"
- `vocabulary`: 3-8 values when possible. Labels are 1-3 common words and distinguishable by a first-time learner. One "other" or "not applicable" value is allowed.
- `definition` and value definitions: one plain sentence, roughly 15 words or fewer.
- `guidelines`: annotator-facing edge cases, tie-breakers, boundary conditions, and label-change triggers.

DEPTH RULE
A proposal is expected on every call. When broad dimensions are covered, go finer-grained within the learner's situation rather than refusing — narrower factors are acceptable as long as every value completes "Show me the segments for when ___" and the factor is not an outright duplicate. Only if every conceivable candidate would duplicate an existing factor, return an empty candidate list and state which candidates you considered and why each is a duplicate.

EXAMPLE (different domain: pottery)
BAD: title="Prehension Patterns", user_question="How is the object grasped?", vocabulary=["precision pinch", "power grasp"].
GOOD: title="Clay Stage", user_question="How firm is your clay right now?", vocabulary=["wet", "leather-hard", "bone-dry", "fired"].

When accepted/rejected facet combinations are provided, use them as feedback:
- Accepted combinations show useful non-redundant dimensions.
- Rejected combinations usually mean the combination was redundant with an accepted one; do not treat every included facet as bad.
- Prefer a learner-situation factor over a technical variant of a rejected combination.

For each proposed contextual factor, provide:
- `title`: short name (1-3 words)
- `title_plural`: plural form of the title
- `user_question`: a learner-facing question used to choose the factor value
- `scope`: one of "global", "mid-level", or "fine-grained"
- `definition`: brief description of what this factor captures
- `justification`: include the two concrete learners, plus why it is distinct from and not predictable from existing factors
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


_SECTION_SCHEMAS = """- Accepted and rejected facet combinations:
{factor_combinations}
"""

_SECTION_HISTORY = """- Previously tried factors that were not kept:
Do not re-propose these factors or close paraphrases of them.
{history}
"""

_REMINDER = """REMINDER: propose exactly one factor — the best remaining candidate, going finer-grained within the learner's situation when broad dimensions are covered; every value must complete "Show me the segments for when ___"; vocabulary uses 3-8 plain-word values when possible. An empty list is reserved for the case where every candidate would outright duplicate an existing factor.
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
            scope=candidate["scope"],
            definition=candidate["definition"],
            justification=candidate["justification"],
            guidelines=candidate["guidelines"],
            vocabulary=vocabulary,
        ))
    return facets
