"""
Prompt templates for VLM-based knowledge extraction and facet discovery.

Each prompt module provides form_*/process_* function pairs:
- form_*_request() - Builds messages + response_format for the LLM
- process_*_response() - Parses LLM response into application objects

Naming convention:
- ki_* : Knowledge item prompts
- fa_* : Facet prompts

Formatting helpers are in formatters.py.
"""

from . import ki_atomic_extraction
from . import ki_step_aggregation
from . import ki_step_extraction
from . import ki_tutorial_summary
from . import fa_facet_discovery
from . import fa_facet_revision

__all__ = [
    "ki_atomic_extraction",
    "ki_step_aggregation",
    "ki_step_extraction",
    "ki_tutorial_summary",
    "fa_facet_discovery",
    "fa_facet_revision",
]
