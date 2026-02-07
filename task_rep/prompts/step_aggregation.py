"""
Prompts for aggregating discovered steps into a canonical taxonomy.

Takes steps discovered from multiple videos and creates a unified,
consistent taxonomy of step names and definitions.
"""

from typing import Any, Dict, List
from pydantic import BaseModel, ConfigDict, Field

from task_rep.utils.llm import MessageContent

class TaxonomyItem(BaseModel):
    """Item in the step taxonomy."""
    model_config = ConfigDict(extra='forbid')

    name: str = Field(description="Canonical name of the step")
    definition: str = Field(description="Defintion of the step that describes what qualifies as this step")
    source_variants: List[str] = Field(description="List of original step names that were merged into this canonical step (for traceability)")

class StepTaxonomy(BaseModel):
    """Aggregated taxonomy of canonical step names across all videos."""
    model_config = ConfigDict(extra='forbid')

    steps: List[TaxonomyItem] = Field(description="List of canonical step names and definitions")


SYSTEM_MESSAGE = """You are an expert at analyzing procedural knowledge and creating step taxonomies for the task: {task}. You excel at identifying similar or duplicate steps across multiple tutorials, understanding naming variations, and creating clear, consistent canonical names with comprehensive definitions. You understand logical ordering of procedural steps and can recognize when steps from different videos represent the same action versus genuinely distinct actions."""

USER_MESSAGE = """You are given steps discovered from multiple tutorial videos.

GOAL
Your goal is to produce a **step taxonomy**: a unified inventory of all distinct steps that *could* appear in any valid procedure for this task. This is not a single linear procedure — different tutorials may include, omit, or substitute steps. The taxonomy should cover the full space of possible steps observed across all videos.

REQUIREMENTS:
1. **Merge equivalent steps.** If two or more steps from *different* videos describe the same action (even if worded differently), collapse them into a single canonical step.
2. **Keep genuinely distinct steps separate.** If two steps appear *within the same video* as separate actions, they are distinct and should remain separate entries — even if their names sound similar.
3. **Enforce consistent granularity.** Every step in the taxonomy should represent a similar "size" of action, roughly the kind of segment you'd label in a video (a single coherent physical or conceptual action lasting roughly 30-120 seconds). Avoid mixing high-level phases (e.g., "Prepare the surface") with fine-grained micro-actions (e.g., "Pick up sandpaper"). If you encounter steps at mismatched levels, either split coarse steps or merge overly fine ones so the taxonomy is uniform.
4. **Use clear canonical names and definitions.** Each step should have a concise, verb-phrase canonical name (e.g., "Sand the edges", "Apply primer coat") and a definition (1-3 sentences) describing what the step involves, broad enough to cover the variations seen across videos but specific enough to distinguish it from other steps.

OUTPUT:
Return a list of step names and definitions.
The `source_variants` field should list the original step names that were merged into this canonical step (for traceability).

Discovered steps (grouped by video):
```
{steps_text}
```
"""

def form_step_aggregation_request(
    task: str,
    discovered_steps: List[Dict[str, Any]],
    **kwargs,    
) -> Dict[str, Any]:


    # Group steps by video for better presentation
    steps_by_video = {}
    for step in discovered_steps:
        video_id = step.get("video_id", "unknown")
        video_title = step.get("video_title", "Unknown")
        if video_id not in steps_by_video:
            steps_by_video[video_id] = {
                "title": video_title,
                "steps": [],
            }
        steps_by_video[video_id]["steps"].append(step)

    # Format steps grouped by video
    steps_text_parts = []
    for video_idx, video_data in enumerate(steps_by_video.values()):
        steps_text_parts.append(f"\nVideo {video_idx + 1}: {video_data['title']}")
        for idx, step in enumerate(video_data["steps"]):
            steps_text_parts.append(f"\t- ({video_idx}, {idx}): {step.get('name')}: {step.get('content')}")

    steps_text = "\n".join(steps_text_parts)

    messages = [
        {
            "role": "system",
            "content": MessageContent(content=[
                {"type": "text", "text": SYSTEM_MESSAGE.format(task=task)}
            ])
        },
        {
            "role": "user",
            "content": MessageContent(content=[
                {"type": "text", "text": USER_MESSAGE.format(steps_text=steps_text)}
            ])
        }
    ]
    return {
        "messages": messages,
        "response_format": StepTaxonomy,
    }

def process_step_aggregation_response(
    response: Dict[str, Any],
    **kwargs,
) -> List[Dict[str, Any]]:
    """Process response from step aggregation."""
    result = response.get("response", {})
    steps = result.get("steps", [])

    taxonomy = []
    for step in steps:
        name = step.get("name")
        definition = step.get("definition")
        taxonomy.append({
            "name": name,
            "definition": definition,
        })
    return taxonomy