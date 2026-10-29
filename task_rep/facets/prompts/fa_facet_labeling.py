"""
Prompts for labeling knowledge items with facet vocabulary labels.

One request per (facet, video) pair: assigns a label from the facet's
vocabulary to each knowledge item in that video.
"""

import difflib
import logging
from typing import Any, Dict, List, Optional

from utils.llm import MessageContent

from .fa_pydantic import Segmentation
from .formatters import (
    facet_to_message_content,
    knowledge_items_to_message_content,
    build_item_id_map,
    reverse_id_map,
)

from facets.schema import Facet
from knowledge_items.schema import KnowledgeItem

logger = logging.getLogger(__name__)


LABELING_SYSTEM_PROMPT = """You are an expert at analyzing tutorial content and labeling tutorial segments with contextual factors related to the task: {task}. Inputs are structured with XML tags. Use tag names and id attributes to identify objects and boundaries."""

LABELING_USER_PROMPT = """You are given a contextual factor with its vocabulary, and the segments of one tutorial in order. Assign each segment one value from the factor's vocabulary.

HOW TO LABEL, BY THE FACTOR'S KIND
- situation: the value describes the learner's whole attempt (what they are making, with what). Use everything this tutorial shows, including other segments, and give each segment the value that holds for the attempt at that point; it is usually the same for the whole tutorial. Use "unspecified" only if the tutorial never reveals the value.
- progress: give the value for where in the task this segment is.
- moment: give this segment's own value; use "not applicable" when the segment is not part of the step the factor is about.
- If the factor has no kind, decide from its definition which kind it resembles.

REQUIREMENTS
- Exactly one value per segment, written exactly as in the <name> tag.
- Follow the factor's definition, value definitions, and guidelines.
- If no value fits and the vocabulary has no "unspecified" or "not applicable" value, choose the value that fits best.

INPUTS
- Contextual factor:
{facet}
- Tutorial segments from this video, in order:
{video_items}"""


def form_labeling_request(
    task: str,
    facet: Facet,
    video_items: List[KnowledgeItem],
    **kwargs,
) -> Dict[str, Any]:
    """Form request to label items from one video with one facet."""
    id_map = build_item_id_map(video_items)
    messages = [
        {
            "role": "system",
            "content": MessageContent.from_template(LABELING_SYSTEM_PROMPT, task=task),
        },
        {
            "role": "user",
            "content": MessageContent.from_template(
                LABELING_USER_PROMPT,
                facet=facet_to_message_content(facet, "F"),
                video_items=knowledge_items_to_message_content(video_items, id_map),
            ),
        },
    ]
    return {
        "messages": messages,
        "response_format": Segmentation,
    }


def _fuzzy_match_label(label: str, valid_labels: set[str]) -> Optional[str]:
    """Try to recover a truncated/near-miss label via prefix then difflib matching."""
    # Prefix match: returned label is a strict prefix of exactly one valid label
    prefix_matches = [v for v in valid_labels if v.startswith(label) and v != label]
    if len(prefix_matches) == 1:
        return prefix_matches[0]

    # difflib closest match (cutoff=0.85 avoids false positives across short labels)
    close = difflib.get_close_matches(label, valid_labels, n=1, cutoff=0.85)
    if close:
        return close[0]

    return None

def process_labeling_response(
    response: Dict[str, Any],
    facet: Facet,
    video_items: List[KnowledgeItem],
    **kwargs,
) -> List[Dict[str, str]]:
    """Process response from labeling request.

    Remaps temp IDs (I1, I2, ...) back to real item IDs.
    Logs errors for: hallucinated item IDs, hallucinated labels, and unlabeled items.

    Returns:
        List of dicts: [{"item_id": str, "label": str}]
    """
    id_map = build_item_id_map(video_items)
    rev_map = reverse_id_map(id_map)  # temp_id → real_id
    valid_labels = {lbl.label.lower() for lbl in facet.vocabulary}

    result = response.get("response", {})
    segmentation = result.get("segmentation", [])

    entries = []
    labeled_temp_ids = set()
    for entry in segmentation:
        temp_id = entry.get("segment_id", "").strip()
        label = entry.get("label", "").strip().lower()

        real_id = rev_map.get(temp_id)
        if real_id is None:
            logger.error(
                "Labeling hallucinated segment_id %r (facet=%r). Dropping.",
                temp_id, facet.title,
            )
            continue

        if label not in valid_labels:
            recovered = _fuzzy_match_label(label, valid_labels)
            if recovered:
                logger.warning(
                    "Labeling fuzzy-matched label %r → %r for segment %r (facet=%r).",
                    label, recovered, temp_id, facet.title,
                )
                label = recovered
            else:
                logger.error(
                    "Labeling hallucinated label %r for segment %r (facet=%r). Dropping.",
                    label, temp_id, facet.title,
                )
                continue

        labeled_temp_ids.add(temp_id)
        entries.append({"item_id": real_id, "label": label})

    unlabeled = set(id_map.values()) - labeled_temp_ids
    if unlabeled:
        logger.error(
            "Labeling skipped %d/%d items for facet=%r: %s",
            len(unlabeled), len(video_items), facet.title, sorted(unlabeled),
        )

    return entries
