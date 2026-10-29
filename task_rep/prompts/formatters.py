# Formatting helpers for facet-related prompts

from typing import List, Dict, Any

from task_rep.utils.llm import MessageContent

def taxonomy_to_str(taxonomy: List[Dict[str, Any]]) -> str:
    """Format taxonomy as a string for prompts."""
    if not taxonomy:
        return "(No taxonomy provided)"

    lines = []
    for item in taxonomy:
        lines.append(f"- [{item.get('name')}] {item.get('definition')}")
    return "\n".join(lines)

def dict_mc_to_str(dict_mc: Dict[str, MessageContent]) -> MessageContent:
    """Format dictionary of MessageContent as a string for prompts."""
    content = []
    for key, value in dict_mc.items():
        content.extend([
            {"type": "text", "text": f"- [{key}]"}, *value.to_content()
        ])
    return MessageContent(content=content)

def list_mc_to_str(list_mc: List[MessageContent]) -> MessageContent:
    """Format list of MessageContent as a string for prompts."""
    content = []
    for value in list_mc:
        content.extend([*value.to_content()])
    return MessageContent(content=content)