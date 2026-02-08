"""
Prompt templates for VLM-based knowledge extraction.

Each prompt module provides:
- get_system_message() - Returns the system message for the LLM
- get_instruction() - Returns the user instruction template
"""

from . import atomic_extraction
from . import step_aggregation
from . import step_extraction
from . import tutorial_summary
from . import facet_vocabulary
from . import facet_labeling

__all__ = [
    "atomic_extraction",
    "step_aggregation",
    "step_extraction",
    "tutorial_summary",
    "facet_vocabulary",
    "facet_labeling",
    "knowledge_items_to_str",
    "facets_to_str",
    "facet_labels_to_str",
]
