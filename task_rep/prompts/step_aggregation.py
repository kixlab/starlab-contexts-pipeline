"""
Prompts for aggregating discovered steps into a canonical taxonomy.

Takes steps discovered from multiple videos and creates a unified,
consistent taxonomy of step names.
"""


def get_system_message() -> str:
    """Get the system message for step taxonomy aggregation."""
    return """You are an expert at analyzing procedural knowledge and creating taxonomies. You excel at identifying similar or duplicate steps across multiple tutorials, understanding naming variations, and creating clear, consistent canonical names. You understand logical ordering of procedural steps and can recognize when similar steps from different videos represent the same procedure versus when they are genuinely different steps."""


def get_instruction(discovered_steps: list[dict]) -> str:
    """
    Get user instruction for taxonomy aggregation.

    Args:
        discovered_steps: List of dicts with 'name', 'content', 'video_id', 'video_title'

    Returns:
        Instruction text for the user message
    """
    # Group steps by video for better presentation
    steps_by_video = {}
    for step in discovered_steps:
        video_id = step.get("video_id", "unknown")
        video_title = step.get("video_title", "Unknown")
        if video_id not in steps_by_video:
            steps_by_video[video_id] = {
                "title": video_title,
                "steps": []
            }
        steps_by_video[video_id]["steps"].append(step)

    # Format steps grouped by video
    steps_text_parts = []
    for video_idx, video_data in enumerate(steps_by_video.values()):
        steps_text_parts.append(f"\nVideo {video_idx + 1}: {video_data['title']}")
        for step in video_data["steps"]:
            steps_text_parts.append(f"  - {step.get('name')}: {step.get('content')}")

    steps_text = "\n".join(steps_text_parts)

    return f"""Here are steps discovered from multiple tutorial videos:

{steps_text}

Create a canonical taxonomy by:
1. Merging similar/duplicate steps across different videos (steps with similar names/descriptions that represent the same procedure)
2. Keeping distinct steps that appear in the same video as separate (they are different procedures)
3. Using clear, consistent naming for the canonical step names
4. Ordering logically (preparation → main procedure → finishing)

Return a list of canonical step names (just the names, not descriptions)."""