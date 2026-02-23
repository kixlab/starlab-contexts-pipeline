"""
Facet discovery and annotation module.

This module provides data structures for facets and annotations, and
VLM-based operations for discovering, revising, building vocabulary for,
and labeling with facets.

Public API:
- Facet, FacetLabel, AnnotationTable: Core data structures
- VLMFacetExtractor: Batch-first VLM-based facet discovery and revision
- VLMFacetVocabularyBuilder: VLM-based vocabulary construction
- VLMFacetLabeler: VLM-based item labeling
"""

from .schema import (
    Facet,
    FacetLabel,
    AnnotationTable,
)
from .discovery import (
    VLMFacetExtractor,
)
from .vocabulary import (
    VLMFacetVocabularyBuilder,
)
from .label import (
    VLMFacetLabeler,
)