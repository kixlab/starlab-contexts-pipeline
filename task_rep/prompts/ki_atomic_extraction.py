"""
Prompts for atomic information extraction from tutorial videos.

Atomic information refers to individual pieces of knowledge that can be
classified into a taxonomy (e.g., methods, tips, context, explanations).
"""

from typing import List, Dict, Any

from utils.llm import MessageContent
from .formatters import taxonomy_to_str

from .ki_pydantic import AtomicExtractionResult

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
    tutorial_content: MessageContent,
    information_taxonomy: List[Dict[str, Any]],
    **kwargs,
) -> Dict[str, Any]:
    """Form request to extract atomic information from a tutorial."""
    information_taxonomy_str = taxonomy_to_str(information_taxonomy)
    messages = [
        {
            "role": "system",
            "content": MessageContent.from_template(
                SYSTEM_MESSAGE,
                task=task
            )
        },
        {
            "role": "user",
            "content": MessageContent.from_template(
                USER_MESSAGE,
                taxonomy=information_taxonomy_str,
                tutorial=tutorial_content
            )
        }
    ]
    return {
        "messages": messages,
        "response_format": AtomicExtractionResult,
    }

def process_atomic_extraction_response(
    response: Dict[str, Any],
    information_taxonomy: List[Dict[str, Any]],
    **kwargs,
) -> List[Dict[str, Any]]:
    """Process response from atomic extraction from a tutorial."""
    result = response.get("response", {})
    items = result.get("items", [])
    atomic_items = []

    valid_atomic_types = [info.get('name') for info in information_taxonomy]
    for item in items:
        atomic_type = item.get("atomic_type")
        if atomic_type not in valid_atomic_types:
            continue
        content = item.get("content")
        start = item.get("start")
        end = item.get("end")
        atomic_items.append({
            "atomic_type": atomic_type,
            "content": content,
            "start": float(start),
            "end": float(end),
        })
    return atomic_items