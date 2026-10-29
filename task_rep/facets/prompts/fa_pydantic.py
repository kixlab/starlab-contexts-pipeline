from pydantic import BaseModel, ConfigDict, Field
from typing import List

class SegmentLabel(BaseModel):
    """A label in a facet's vocabulary."""
    model_config = ConfigDict(extra='forbid')

    label: str = Field(description="Label text")
    definition: str = Field(description="Definition of this label")

#----------------------------------------

class RevisedFacetCandidate(BaseModel):
    """A revised facet candidate."""
    model_config = ConfigDict(extra='forbid')

    title: str = Field(description="Factor title")
    title_plural: str = Field(description="Plural form")
    scope: str = Field(description="global, mid-level, or fine-grained")
    justification: str = Field(
        description="Brief justification of changes and why this facet is (1) data-grounded, (2) task-relevant, (3) monolithic, (4) piecewise-stable, and (5) target-independent."
    )
    definition: str = Field(description="Updated definition")
    guidelines: List[str] = Field(description="Updated instructions for labeling")
    segment_labels: List[SegmentLabel] = Field(description="Updated vocabulary")

class FacetRevisionResult(BaseModel):
    """Result of facet revision."""
    model_config = ConfigDict(extra='forbid')

    candidates: List[RevisedFacetCandidate] = Field(description="Revised facets")
    changes: str = Field(description="Summary of what was changed and why. If no changes are proposed, a brief explanation of why.")

#----------------------------------------

class SegmentationEntry(BaseModel):
    """Single item-label assignment in a segmentation."""
    model_config = ConfigDict(extra='forbid')

    item_id: str = Field(description="ID of the knowledge item")
    label: str = Field(description="Label name from the vocabulary — the exact text inside the <name> tag")

class Segmentation(BaseModel):
    """A segmentation of items by a single facet."""
    model_config = ConfigDict(extra='forbid')

    segmentation: List[SegmentationEntry] = Field(description="Label assignment for each item")


#----------------------------------------

class TargetAssignment(BaseModel):
    """Label assignment for a target item."""
    model_config = ConfigDict(extra='forbid')

    item_id: str = Field(description="ID of the target information unit")
    label: str = Field(description="Assigned label from the vocabulary")
    brief_reason: str = Field(
        description="One-sentence justification for why this label applies "
        "based on the surrounding context, not the item content itself"
    )


class AnchorExample(BaseModel):
    """A non-target item demonstrating that the factor applies broadly."""
    model_config = ConfigDict(extra='forbid')

    item_id: str = Field(description="ID of a non-target item from the tutorial")
    label: str = Field(description="Label that would apply to this item's context")
    brief_reason: str = Field(
        description="Why this anchor demonstrates the factor's breadth"
    )

#----------------------------------------

class FacetCandidate(BaseModel):
    """A proposed facet candidate with vocabulary and segmentations."""
    model_config = ConfigDict(extra='forbid')

    title: str = Field(description="Short name (2-3 words)")
    title_plural: str = Field(description="Plural form of the title")
    scope: str = Field(description="global, mid-level, or fine-grained")
    justification: str = Field(
        description="Why this facet was chosen: (1) satisfies data-grounded, task-relevant, monolithic, piecewise-stable, and target-independent, (2) how it separates colliding items, (3) why it is distinct from previous factors"
    )
    definition: str = Field(description="What this facet captures")
    guidelines: List[str] = Field(description="Instructions for labeling")
    segment_labels: list[SegmentLabel] = Field(description="Full vocabulary")
    target_assignments: list[TargetAssignment] = Field(
        description="Label for each target item under this factor"
    )
    anchor_examples: list[AnchorExample] = Field(
        min_length=2,
        max_length=4,
        description="Non-target items showing the factor applies broadly "
        "across the tutorial with different labels used",
    )


class FacetDiscoveryResult(BaseModel):
    """Result of facet discovery."""
    model_config = ConfigDict(extra='forbid')

    candidates: list[FacetCandidate] = Field(
        description="Proposed facet candidates. Empty list if no valid factors found."
    )
    refusal_reason: str = Field(
        description="If no candidates are proposed, a brief explanation of why the target items cannot be meaningfully distinguished by contextual factors. Otherwise, leave it empty.",
    )

#----------------------------------------
# Facet Initialization (bootstrap)
#----------------------------------------

class FacetInitCandidate(BaseModel):
    """A facet candidate proposed during initialization (no target assignments)."""
    model_config = ConfigDict(extra='forbid')

    title: str = Field(description="Short name (2-3 words)")
    title_plural: str = Field(description="Plural form of the title")
    scope: str = Field(description="global, mid-level, or fine-grained")
    justification: str = Field(
        description="Why this facet was chosen: (1) satisfies data-grounded, task-relevant, monolithic, piecewise-stable, and target-independent, and (2) why it is distinct from previous factors"
    )
    definition: str = Field(description="What this facet captures")
    guidelines: List[str] = Field(description="Instructions for labeling")
    segment_labels: list[SegmentLabel] = Field(description="Full vocabulary")


class FacetInitResult(BaseModel):
    """Result of facet initialization."""
    model_config = ConfigDict(extra='forbid')

    candidates: list[FacetInitCandidate] = Field(
        description="Proposed facet candidates for bootstrapping the algorithm."
    )

#----------------------------------------
# Vocabulary Building
#----------------------------------------

class ProposedLabel(BaseModel):
    """A single proposed vocabulary label for a facet."""
    model_config = ConfigDict(extra='forbid')

    label: str = Field(description="Short label text (2-5 words)")
    definition: str = Field(description="Definition of what this label means in the context of the facet")

class VocabularyProposalResult(BaseModel):
    """Result of per-video vocabulary proposal for a single facet."""
    model_config = ConfigDict(extra='forbid')

    proposed_labels: List[ProposedLabel] = Field(description="Proposed vocabulary labels")

#----------------------------------------

class ConsolidatedLabel(BaseModel):
    """A single consolidated vocabulary label after deduplication."""
    model_config = ConfigDict(extra='forbid')

    label: str = Field(description="Canonical label text")
    definition: str = Field(description="Refined definition")
    source_labels: List[str] = Field(description="Original proposed labels that were merged into this one")

class VocabularyConsolidationResult(BaseModel):
    """Result of consolidating vocabulary for a single facet."""
    model_config = ConfigDict(extra='forbid')

    consolidated_labels: List[ConsolidatedLabel] = Field(description="Final deduplicated vocabulary labels")
    consolidation_notes: str = Field(description="Brief explanation of merging decisions")
