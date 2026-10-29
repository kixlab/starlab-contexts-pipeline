"""
Prompts for atomic information extraction from tutorial videos.

Atomic information refers to individual pieces of knowledge that can be
classified into a taxonomy (e.g., methods, tips, context, explanations).
"""

import logging
from typing import List, Dict, Any

from utils import MessageContent

from .formatters import taxonomy_to_message_content, video_to_message_content
from .ki_pydantic import AtomicExtractionResult

from knowledge_items.schema import KnowledgeItem, KnowledgeItemType, KnowledgeSource, Video
from knowledge_items.taxonomy import AtomicType
from knowledge_items.chunk_sampling import ChunkConfig

logger = logging.getLogger(__name__)

SYSTEM_MESSAGE = """You are an expert at analyzing tutorial videos and extracting atomic pieces of information about the task: {task}. You excel at identifying atomic information from both visual and audio content. You understand how to classify information using taxonomies and can accurately identify the start and end times of each piece of information."""

USER_MESSAGE = """You are given a taxonomy of atomic information and a tutorial with temporally aligned frames and transcript. Extract ALL atomic information pieces that are relevant to the task from the tutorial.

PROCEDURE:
1. Detect task-relevant information: Parse the transcript clause-by-clause and create one piece for each indivisible action, object, reason, or tip.
    - Indivisible = removing any word breaks the meaning.
    - If a sentence contains multiple actions (`whisk, then fold`) or multiple rationales, split them.
    - Rewrite each piece so it is understandable standing alone—no dangling `it`,`they`, `this step`, etc.
Examples:
    `Whisk the eggs for 30 s, then fold in the flour.` -> `Whisk the eggs for 30 seconds.` (Method - Instruction), `Fold the eggs in the flour.` (Method - Instruction)
    `Let the paint dry so it won't smudge.` -> `Let the paint dry.` (Method - Instruction), `Drying the paint prevents smudging.` (Explanation - Justification)
    `Add the beaten eggs to the mixture and mix well.` -> `Add the beaten eggs to the mixture.` (Method - Instruction), `Mix beaten eggs well.` (Method - Instruction)
    `Add 1 cup of white chocolate chips and stir until thoroughly combined.` -> `Add 1 cup of white chocolate chips.` (Method - Instruction), `Stir white chocolate chips until thoroughly combined.` (Method - Instruction)
2. Assign `atomic_type` to each piece according to taxonomy below.
3. Provide timestamps for each piece by copying the exact start and end (in seconds) that bracket the source text of the item.
    - If no timing metadata exists, output null for both.

INPUT:
- Taxonomy of atomic information:
{taxonomy}
- Tutorial:
{tutorial}

OUTPUT:
Return a list of atomic information pieces, classified based on the taxonomy."""


def form_atomic_extraction_request(
    task: str,
    video: Video,
    use_vision: bool,
    chunk_config: ChunkConfig,
    information_taxonomy: List[Dict[str, Any]],
    **kwargs,
) -> Dict[str, Any]:
    """Form request to extract atomic information from a tutorial."""
    taxonomy_content = taxonomy_to_message_content(information_taxonomy)
    tutorial_content = video_to_message_content(video, use_vision, chunk_config)
    messages = [
        {
            "role": "system",
            "content": MessageContent.from_template(SYSTEM_MESSAGE, task=task),
        },
        {
            "role": "user",
            "content": MessageContent.from_template(
                USER_MESSAGE,
                taxonomy=taxonomy_content,
                tutorial=tutorial_content,
            ),
        },
    ]
    return {
        "messages": messages,
        "response_format": AtomicExtractionResult,
    }


def process_atomic_extraction_response(
    response: Dict[str, Any],
    video: Video,
    information_taxonomy: List[Dict[str, Any]],
    **kwargs,
) -> List[KnowledgeItem]:
    """Process response from atomic extraction, returning typed KnowledgeItems."""
    result = response.get("response", {})
    items = result.get("items", [])

    valid_atomic_types = {info.get('name') for info in information_taxonomy}
    knowledge_items = []
    for idx, item in enumerate(items):
        atomic_type_str = item.get("atomic_type")
        if atomic_type_str not in valid_atomic_types:
            continue
        try:
            atomic_type = AtomicType(atomic_type_str)
        except ValueError:
            logger.warning(f"Invalid atomic type: {atomic_type_str}! Skipping...")
            continue
        content = item.get("content")
        start = item.get("start")
        end = item.get("end")
        knowledge_items.append(KnowledgeItem(
            id=f"atomic:{atomic_type_str}-{idx}:{video.id}",
            type=KnowledgeItemType.ATOMIC,
            text=content,
            source=KnowledgeSource(
                video_id=video.id,
                source_type="video",
                local_index=idx,
                time_range=(start, end),
            ),
            subtype=atomic_type,
        ))
    return knowledge_items
