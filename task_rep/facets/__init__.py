"""
Facet discovery and annotation module.

This module provides functionality for discovering contextual factors (facets)
that discriminate knowledge items, and for annotating items with facet labels.

Public API:
- Facet, FacetLabel: Data structures for facets
- VLMFacetExtractor: Batch-first VLM-based facet discovery and revision
"""

from .schema import Facet, FacetLabel, AnnotationTable
from .discovery import VLMFacetExtractor
from .vocabulary import VLMFacetVocabularyBuilder
from .label import VLMFacetLabeler

__all__ = [
    # Data structures
    "Facet",
    "FacetLabel",
    "AnnotationTable",
    # Discovery
    "VLMFacetExtractor",
    # Vocabulary
    "VLMFacetVocabularyBuilder",
    # Labeling
    "VLMFacetLabeler",
]
