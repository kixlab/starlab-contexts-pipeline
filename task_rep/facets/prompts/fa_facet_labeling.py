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


LABELING_SYSTEM_PROMPT = """You are an expert at analyzing tutorial content and labeling knowledge items with contextual factors related to the task: {task}. Inputs are structured with XML tags — use the tag name and id attribute to identify each object and its boundaries."""

LABELING_USER_PROMPT = """You are given a contextual factor (facet) with its vocabulary, and knowledge items extracted from a tutorial video. For each knowledge item, assign the single most appropriate label from the facet's vocabulary.

DEFINITIONS
- A valid contextual factor is a variable describing the procedural situation at a given moment or span. It must be:
    - Domain-grounded: anchored in the task-world (e.g., procedural stage, material state, tool state, object configuration, environmental condition, operating mode, verification state, service context) — NOT in media, narration, editing, presentation, or social framing.
    - Consequential: changes what actions, constraints, interpretations, checks, or information apply.
    - Monolithic: captures one coherent semantic concept, not a bundle.
    - Piecewise-stable: remains constant over meaningful spans; changes at identifiable boundaries.
    - Target-independent: determined by the broader procedural context; removing the target item would not change the factor's value.
- A vocabulary label is one possible value a facet can take.

REQUIREMENTS
- Assign exactly one label to each knowledge item.
- The label MUST be one of the labels in the facet's vocabulary (use the exact text inside the <name> tag, not the definition text).
- If none of the labels clearly apply, choose the closest match.
- Every item must receive a label — do not skip any items.

INPUTS
- Facet:
{facet}
- Knowledge items from this video:
{video_items}
"""


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
        temp_id = entry.get("item_id", "").strip()
        label = entry.get("label", "").strip().lower()

        real_id = rev_map.get(temp_id)
        if real_id is None:
            logger.error(
                "Labeling hallucinated item_id %r (facet=%r). Dropping.",
                temp_id, facet.title,
            )
            continue

        if label not in valid_labels:
            recovered = _fuzzy_match_label(label, valid_labels)
            if recovered:
                logger.warning(
                    "Labeling fuzzy-matched label %r → %r for item %r (facet=%r).",
                    label, recovered, temp_id, facet.title,
                )
                label = recovered
            else:
                logger.error(
                    "Labeling hallucinated label %r for item %r (facet=%r). Dropping.",
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
