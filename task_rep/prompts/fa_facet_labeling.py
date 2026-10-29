"""
Prompts for labeling knowledge items with facet vocabulary labels.

One request per (facet, video) pair: assigns a label from the facet's
vocabulary to each knowledge item in that video.
"""

from typing import Any, Dict, List

from utils.llm import MessageContent

from .fa_pydantic import Segmentation
from .formatters import list_mc_to_str


LABELING_SYSTEM_PROMPT = """You are an expert at analyzing tutorial content and labeling knowledge items with contextual factors related to the task: {task}."""

LABELING_USER_PROMPT = """You are given a contextual factor (facet) with its vocabulary, and knowledge items extracted from a tutorial video. For each knowledge item, assign the single most appropriate label from the facet's vocabulary.

DEFINITIONS
- A contextual factor (facet) is a piecewise-stable, target-independent, task-relevant dimension that discriminates knowledge items.
- A vocabulary label is one possible value a facet can take.

REQUIREMENTS
- Assign exactly one label to each knowledge item.
- The label MUST be one of the labels in the facet's vocabulary (use the exact label text).
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
    facet: MessageContent,
    video_items: List[MessageContent],
    **kwargs,
) -> Dict[str, Any]:
    """Form request to label items from one video with one facet."""
    video_items_str = list_mc_to_str(video_items)

    messages = [
        {
            "role": "system",
            "content": MessageContent.from_template(
                LABELING_SYSTEM_PROMPT,
                task=task,
            )
        },
        {
            "role": "user",
            "content": MessageContent.from_template(
                LABELING_USER_PROMPT,
                facet=facet,
                video_items=video_items_str,
            )
        }
    ]
    return {
        "messages": messages,
        "response_format": Segmentation,
    }


def process_labeling_response(
    response: Dict[str, Any],
    **kwargs,
) -> List[Dict[str, str]]:
    """Process response from labeling request.

    Returns:
        List of dicts: [{"item_id": str, "label": str}]
    """
    result = response.get("response", {})
    segmentation = result.get("segmentation", [])

    return [
        {
            "item_id": entry.get("item_id", "").strip(),
            "label": entry.get("label", "").strip(),
        }
        for entry in segmentation
    ]
