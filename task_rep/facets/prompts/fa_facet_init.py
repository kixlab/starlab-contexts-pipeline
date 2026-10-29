"""
Prompts for bootstrapping the iterative algorithm with common-sense contextual factors.

Unlike discovery (which resolves specific collisions), initialization asks the
LLM to propose N broad facets upfront given the tutorials.
"""

import logging
from typing import Any, Dict, List

from utils.llm import MessageContent
from utils import random_uid

from .fa_criteria import FACTOR_CRITERIA
from .fa_pydantic import ContextualFactorInitResult
from .fa_system import system_message

from facets.schema import Facet, FacetLabel
from knowledge_items.schema import KnowledgeItem

logger = logging.getLogger(__name__)


USER_PROMPT = """Propose exactly {num_facets} contextual factors that together help learners find the tutorial segments that fit them. Cover situations, progress, and moments, and start with the broadest factors.

""" + FACTOR_CRITERIA + """

REMINDER: propose exactly {num_facets} distinct factors; every value must complete "Show me the segments for when ___"; prefer factors that apply across most tutorials.
"""


def form_facet_init_request(
    task: str,
    videos: Dict[str, List[KnowledgeItem]],
    num_facets: int,
    **kwargs,
) -> Dict[str, Any]:
    """Form request to initialize facets from tutorials (shown in the system message)."""
    messages = [
        {"role": "system", "content": system_message(task, videos)},
        {
            "role": "user",
            "content": MessageContent.from_template(
                USER_PROMPT, task=task, num_facets=str(num_facets)),
        },
    ]
    return {
        "messages": messages,
        "response_format": ContextualFactorInitResult,
    }


def process_facet_init_response(
    response: Dict[str, Any],
    **kwargs,
) -> List[Facet]:
    """Process facet init response, returning typed Facet objects."""
    result = response.get("response", {})

    refusal_reason = result.get("refusal_reason", "")
    candidates = result.get("candidates", [])

    if not candidates:
        logger.info("Initialization proposal refused: %s", refusal_reason or "(no reason given)")
        return []

    logger.info("Initialization proposed %d facet(s).", len(candidates))

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
