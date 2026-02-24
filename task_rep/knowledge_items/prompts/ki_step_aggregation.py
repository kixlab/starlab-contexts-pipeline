"""
Prompts for aggregating discovered steps into a canonical taxonomy.

Takes steps discovered from multiple videos and creates a unified,
consistent taxonomy of step names and definitions.
"""

from typing import Any, Dict, List

from utils.llm import MessageContent

from .formatters import steps_by_video_to_message_content
from .ki_pydantic import StepTaxonomy

from knowledge_items.schema import KnowledgeItem, Video


SYSTEM_MESSAGE = """You are an expert at analyzing procedural knowledge and creating step taxonomies for the task: {task}. You excel at identifying similar or duplicate steps across multiple tutorials, understanding naming variations, and creating clear, consistent canonical names with comprehensive definitions. You understand logical ordering of procedural steps and can recognize when steps from different videos represent the same action versus genuinely distinct actions."""

USER_MESSAGE = """You are given steps discovered from multiple tutorial videos.

GOAL
Your goal is to produce a **step taxonomy**: a unified inventory of all distinct steps that *could* appear in any valid procedure for this task. This is not a single linear procedure — different tutorials may include, omit, or substitute steps. The taxonomy should cover the full space of possible steps observed across all tutorials.

REQUIREMENTS:
1. **Merge equivalent steps.** If two or more steps from *different* tutorials describe the same action (even if worded differently), collapse them into a single canonical step.
2. **Keep genuinely distinct steps separate.** If two steps appear *within the same tutorial* as separate actions, they are distinct and should remain separate entries — even if their names sound similar.
3. **Enforce consistent granularity.** Every step in the taxonomy should represent a similar "size" of action, roughly the kind of segment you'd label in a tutorial (a single coherent physical or conceptual action lasting roughly 30-120 seconds). Avoid mixing high-level phases (e.g., "Prepare the surface") with fine-grained micro-actions (e.g., "Pick up sandpaper"). If you encounter steps at mismatched levels, either split coarse steps or merge overly fine ones so the taxonomy is uniform.
4. **Use clear canonical names and definitions.** Each step should have a concise, verb-phrase canonical name (e.g., "Sand the edges", "Apply primer coat") and a definition (1-3 sentences) describing what the step involves, broad enough to cover the variations seen across tutorials but specific enough to distinguish it from other steps.

OUTPUT:
Return a list of step names and definitions.
The `source_variants` field should list the original step names that were merged into this canonical step (for traceability).

INPUT:
- Discovered steps (grouped by tutorial):
{steps_by_tutorial}
"""


def form_step_aggregation_request(
    task: str,
    videos: List[Video],
    steps_by_video: Dict[str, List[KnowledgeItem]],
    **kwargs,
) -> Dict[str, Any]:
    """Form request to aggregate discovered steps into a canonical taxonomy."""
    steps_content = steps_by_video_to_message_content(videos, steps_by_video)
    messages = [
        {
            "role": "system",
            "content": MessageContent.from_template(SYSTEM_MESSAGE, task=task),
        },
        {
            "role": "user",
            "content": MessageContent.from_template(
                USER_MESSAGE,
                steps_by_tutorial=steps_content,
            ),
        },
    ]
    return {
        "messages": messages,
        "response_format": StepTaxonomy,
    }


def process_step_aggregation_response(
    response: Dict[str, Any],
    **kwargs,
) -> List[Dict[str, Any]]:
    """Process step aggregation response, returning the canonical taxonomy as dicts."""
    result = response.get("response", {})
    steps = result.get("steps", [])

    taxonomy = []
    for step in steps:
        name = step.get("name", "").strip().lower()
        definition = step.get("definition", "").strip()
        taxonomy.append({"name": name, "definition": definition})
    return taxonomy
