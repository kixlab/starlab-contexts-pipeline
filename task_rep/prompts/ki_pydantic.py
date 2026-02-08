"""
Pydantic models for knowledge items.
"""

from pydantic import BaseModel, ConfigDict, Field
from typing import List


# ATOMIC INFORMATION
class AtomicInformation(BaseModel):
    """Single atomic piece of information with taxonomy classification."""
    model_config = ConfigDict(extra='forbid')

    atomic_type: str = Field(description="Classification from taxonomy (e.g., 'Method - Instruction')")
    content: str = Field(description="Natural text content of this atomic information")
    start: float = Field(description="Start time of the atomic information in the video")
    end: float = Field(description="End time of the atomic information in the video")


class AtomicExtractionResult(BaseModel):
    """Result of extracting all atomic information from a video."""
    model_config = ConfigDict(extra='forbid')

    items: List[AtomicInformation] = Field(description="All atomic information items found in the video")


# STEP TAXONOMY
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

# STEP DISCOVERY AND EXTRACTION
class StepInformation(BaseModel):
    """Information about a single step (used for both discovery and detailed extraction)."""
    model_config = ConfigDict(extra='forbid')

    name: str = Field(description="Name of this step (e.g., 'Prepare rice')")
    content: str = Field(description="Natural text content describing this step")
    start: float = Field(description="Start time of the step in the video")
    end: float = Field(description="End time of the step in the video")

class StepExtractionResult(BaseModel):
    """Result of step extraction from a video."""
    model_config = ConfigDict(extra='forbid')

    steps: List[StepInformation] = Field(description="All steps found in the video")

# TUTORIAL
class TutorialInformation(BaseModel):
    """High-level tutorial summary."""
    model_config = ConfigDict(extra='forbid')

    content: str = Field(description="Comprehensive summary of the entire tutorial")