"""
Formatting helpers for knowledge item prompts.

Converts domain objects (Video, KnowledgeItem) into MessageContent
for inclusion in LLM prompts.
"""

from typing import Any, Dict, List, Optional

from utils import (
    MessageContent,
)

from knowledge_items.schema import KnowledgeItem, Video
from knowledge_items.chunk_sampling import ChunkConfig


def video_to_message_content(
    video: Video,
    use_vision: bool,
    chunk_config: ChunkConfig,
) -> MessageContent:
    """Format a Video as MessageContent for prompt inclusion."""
    if use_vision and not video.frames:
        raise ValueError("When vision is enabled, video must have frames")
    if not video.transcript:
        raise ValueError("Video must have transcript")

    if chunk_config.size == 0:
        segment_groups = [[seg] for seg in video.transcript]
    else:
        segment_groups = video._group_segments(video.transcript, chunk_config.size)

    content: List[Dict[str, Any]] = []
    content.append({"type": "text", "text": f"Title: {video.title}\n"})

    for group in segment_groups:
        group_start = group[0].start
        group_end = group[-1].end
        group_text = " ".join(seg.text for seg in group)
        group_frames = video.sample_frames(
            start_time=group_start,
            end_time=group_end,
            max_frames=chunk_config.frames_per_chunk,
        )

        group_start = int(group_start)
        group_end = int(group_end)

        if group_text or group_frames:
            if group_text:
                content.append({"type": "text", "text": f"[{group_start}-{group_end}] {group_text}\n"})
            if group_frames and use_vision:
                for frame in group_frames:
                    frame_time = int(frame.time)
                    content.append({
                        "type": "text",
                        "text": f"[Visual frame at {frame_time}]",
                    })
                    content.append({"type": "image_path", "path": frame.path})
                    content.append({
                        "type": "text",
                        "text": "\n",
                    })

    return MessageContent(content=content)

_KNOWLEDGE_ITEM_FORMAT = """<item id="{display_id}">
{text}
</item>"""

def format_item(
    item: KnowledgeItem,
    display_id: Optional[str] = None,
    with_type: bool = True,
) -> str:
    """Format a KnowledgeItem as text, using display_id instead of real item.id."""
    did = display_id if display_id is not None else item.id
    text = item.text
    if with_type and item.subtype:
        subtype_val = item.subtype.value if hasattr(item.subtype, "value") else str(item.subtype)
        text = f"({subtype_val}) {text}"
    return _KNOWLEDGE_ITEM_FORMAT.format(display_id=did, text=text)

def knowledge_item_to_message_content(
    item: KnowledgeItem,
    display_id: Optional[str] = None,
    with_type: bool = True,
) -> MessageContent:
    """Format a KnowledgeItem as MessageContent for prompt inclusion."""
    return MessageContent(content=[{"type": "text", "text": format_item(item, display_id, with_type)}])

def _format_taxonomy_item(name: str, definition: str, examples: List[str]) -> str:
    """Format one taxonomy entry as an XML element with optional examples."""
    body = f"Definition: {definition}"
    if examples:
        example_lines = "\n".join(f"- {ex}" for ex in examples)
        body += f"\nExamples:\n{example_lines}"
    return f'<category id="{name}">{body}</category>'

def taxonomy_to_message_content(taxonomy: List[Dict[str, Any]]) -> MessageContent:
    """Format a taxonomy list as MessageContent for prompts.

    Each entry should have 'name', 'definition', and optionally 'examples'
    (list of strings). Entries without 'examples' or with an empty list omit
    the Examples section entirely.
    """
    if not taxonomy:
        return MessageContent(content=[{"type": "text", "text": "(No taxonomy provided)"}])
    lines = [
        _format_taxonomy_item(
            name=item.get("name", ""),
            definition=item.get("definition", ""),
            examples=item.get("examples", []),
        )
        for item in taxonomy
    ]
    return MessageContent(content=[{"type": "text", "text": "\n".join(lines)}])

_STEP_FORMAT = """<step id="{id}" name="{step_id}">{text}</step>"""
def steps_by_video_to_message_content(
    videos: List[Video],
    steps_by_video: Dict[str, List["KnowledgeItem"]],
) -> MessageContent:
    """Format discovered steps grouped by video as MessageContent for prompt inclusion.

    Args:
        videos: Source videos (used to look up titles by ID).
        steps_by_video: Mapping from video_id to the list of discovered step KnowledgeItems.
    """
    video_map = {v.id: v for v in videos}
    lines = []
    for video_idx, (video_id, steps) in enumerate(steps_by_video.items()):
        title = video_map[video_id].title if video_id in video_map else video_id
        lines.append(f"Video {video_idx + 1}: {title}")
        for idx, step in enumerate(steps):
            lines.append(_STEP_FORMAT.format(id=idx + 1, step_id=step.subtype, text=step.text))
    return MessageContent(content=[{"type": "text", "text": "\n".join(lines)}])
