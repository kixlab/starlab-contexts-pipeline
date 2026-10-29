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

from .ki_atomic_extraction import (
    form_atomic_extraction_request,
    process_atomic_extraction_response,
)
from .ki_step_aggregation import (
    form_step_aggregation_request,
    process_step_aggregation_response,
)
from .ki_step_extraction import (
    form_step_extraction_request_discovery,
    process_step_extraction_response_discovery,
    form_step_extraction_request_extraction,
    process_step_extraction_response_extraction,
)
from .ki_tutorial_summary import (
    form_tutorial_summary_request,
    process_tutorial_summary_response,
)
from .fa_facet_discovery import (
    form_facet_discovery_request,
    process_facet_discovery_response,
)
from .fa_facet_revision import (
    form_facet_revision_request,
    process_facet_revision_response,
)

from .fa_facet_vocabulary import (
    form_vocabulary_proposal_request,
    process_vocabulary_proposal_response,
    form_vocabulary_consolidation_request,
    process_vocabulary_consolidation_response,
)

from .fa_facet_labeling import (
    form_labeling_request,
    process_labeling_response,
)