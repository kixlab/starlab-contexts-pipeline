"""
The proposer's role and the tutorial corpus, shared by every proposal prompt.

a0, a1 and a2 send ``system_message`` as their system message, so they see the
same role and the same tutorials; the agent's system prompt is built from
``ROLE``, its working instructions and ``tutorials_text``. What differs between
conditions is only what comes after: the user message (a) or the tool results
(g).
"""

from __future__ import annotations

from typing import Dict, List

from knowledge_items.schema import KnowledgeItem
from utils.llm import MessageContent

from .formatters import corpus_to_message_content

ROLE = """You design user-facing metadata for tutorial videos about: {task}.

You propose contextual factors: questions about a learner's situation, progress, or current step that determine which tutorial segments apply to them.

Your audience is novice and intermediate learners of {task}. Factors become filter menus, timeline labels, and search options. Learners use them to find useful segments before watching, so every factor must be something a learner can answer about themselves without seeing the footage."""

_TUTORIALS = """TUTORIALS
Every segment of the tutorials, grouped by video. Each segment has an id (S1, S2, ...) that later messages refer to{types}.
{tutorials}"""
_TYPES = "; the type in parentheses is the kind of information the segment gives"


def tutorials_text(corpus: Dict[str, List[KnowledgeItem]]) -> str:
    """The corpus block: every evaluated segment with its id, grouped by video."""
    segments = "\n".join(b["text"] for b in corpus_to_message_content(corpus).content)
    typed = any(item.subtype for items in corpus.values() for item in items)
    return _TUTORIALS.format(tutorials=segments, types=_TYPES if typed else "")


def system_message(task: str, corpus: Dict[str, List[KnowledgeItem]]) -> MessageContent:
    """The system message of every pipeline proposal request (a0, a1, a2)."""
    text = ROLE.format(task=task) + "\n\n" + tutorials_text(corpus)
    return MessageContent(content=[{"type": "text", "text": text}])
