from pydantic import BaseModel, ConfigDict, Field
from typing import List, Optional

class VocabularyEntry(BaseModel):
    """A single entry in a contextual factor's vocabulary (one possible value of the factor)."""
    model_config = ConfigDict(extra='forbid')

    label: str = Field(description="Label text")
    definition: str = Field(description="Definition of this label")

#----------------------------------------

class SegmentationEntry(BaseModel):
    """Single tutorial-segment label assignment in a segmentation."""
    model_config = ConfigDict(extra='forbid')

    segment_id: str = Field(description="ID of the tutorial segment")
    label: str = Field(description="Label name from the vocabulary — the exact text inside the <name> tag")

class Segmentation(BaseModel):
    """A segmentation of tutorial segments by a single contextual factor."""
    model_config = ConfigDict(extra='forbid')

    segmentation: List[SegmentationEntry] = Field(description="Label assignment for each segment")


#----------------------------------------

class TargetAssignment(BaseModel):
    """Label assignment for a target tutorial segment."""
    model_config = ConfigDict(extra='forbid')

    segment_id: str = Field(description="ID of the target tutorial segment")
    label: str = Field(description="Assigned label from the vocabulary")
    brief_reason: str = Field(
        description="One-sentence justification for why this label applies based on the proceudral context."
    )


class AnchorExample(BaseModel):
    """A non-target tutorial segment demonstrating that the contextual factor applies broadly."""
    model_config = ConfigDict(extra='forbid')

    segment_id: str = Field(description="ID of a non-target tutorial segment")
    label: str = Field(description="The assigned label for this tutorial segment under the contextual factor")
    brief_reason: str = Field(
        description="Why this anchor demonstrates the contextual factor's breadth"
    )

#----------------------------------------

class ContextualFactorCandidate(BaseModel):
    """A proposed contextual factor."""
    model_config = ConfigDict(extra='forbid')

    title: str = Field(description="Short name (1-3 words)")
    title_plural: str = Field(description="Plural form of the title")
    scope: str = Field(description="global, mid-level, or fine-grained")
    justification: str = Field(
        description="Why this factor was chosen: (1) why it is a valid contextual factor, (2) why it is distinct from previously identified contextual factors"
    )
    definition: str = Field(description="What this factor captures")
    guidelines: List[str] = Field(description="Instructions for labeling tutorial segments under this contextual factor")
    vocabulary: list[VocabularyEntry] = Field(description="Vocabulary for this contextual factor (list of possible label values)")


#----------------------------------------

class ContextualFactorCandidateWithAssignments(ContextualFactorCandidate):
    """A proposed contextual factor with assignments to target tutorial segments."""

    target_assignments: list[TargetAssignment] = Field(
        min_length=1,
        description="Label for each target tutorial segment under this contextual factor"
    )
    anchor_examples: list[AnchorExample] = Field(
        min_length=2,
        max_length=4,
        description="Non-target tutorial segments showing the contextual factor applies broadly, with different labels used",
    )


class ContextualFactorDiscoveryResult(BaseModel):
    """Result of contextual factor discovery."""
    model_config = ConfigDict(extra='forbid')

    candidates: list[ContextualFactorCandidateWithAssignments] = Field(
        min_length=0,
        description="Proposed contextual factors. Empty list if none are valid."
    )
    refusal_reason: str = Field(
        description="If no candidates are proposed, a brief explanation of why the target tutorial segments cannot be meaningfully distinguished by contextual factors. Otherwise, leave it empty.",
    )

#----------------------------------------
# Contextual Factor Revision
#----------------------------------------

class ContextualFactorRevisionResult(BaseModel):
    """Result of contextual factor revision."""
    model_config = ConfigDict(extra='forbid')

    candidates: List[ContextualFactorCandidate] = Field(description="Revised contextual factors")
    changes: str = Field(description="Summary of what was changed and why. If no changes are proposed, a brief explanation of why.")

#----------------------------------------
# Contextual Factor Initialization
#----------------------------------------

class ContextualFactorInitResult(BaseModel):
    """Result of contextual factor proposal."""
    model_config = ConfigDict(extra='forbid')

    candidates: list[ContextualFactorCandidate] = Field(
        min_length=0,
        description="Proposed contextual factors. Empty list if none are valid."
    )
    refusal_reason: str = Field(
        description="If no candidates are proposed, a brief explanation of why. Otherwise, leave it empty.",
    )

#----------------------------------------
# Vocabulary Building
#----------------------------------------

class ProposedLabel(BaseModel):
    """A single proposed vocabulary label for a contextual factor."""
    model_config = ConfigDict(extra='forbid')

    label: str = Field(description="Short label text (1-3 words)")
    definition: str = Field(description="Definition of what this label means in the context of the contextual factor")

class VocabularyProposalResult(BaseModel):
    """Result of per-tutorial vocabulary proposal for a single contextual factor."""
    model_config = ConfigDict(extra='forbid')

    proposed_labels: List[ProposedLabel] = Field(description="Proposed vocabulary labels")

#----------------------------------------

class ConsolidatedLabel(BaseModel):
    """A single consolidated vocabulary label after deduplication."""
    model_config = ConfigDict(extra='forbid')

    label: str = Field(description="Canonical label text")
    definition: str = Field(description="Refined definition")
    source_labels: List[str] = Field(description="Original proposed labels that were merged into this one")

class ContextualFactorConsolidationResult(BaseModel):
    """Result of consolidating a contextual factor's vocabulary.

    The definition and guidelines are realigned only as needed to stay consistent
    with the consolidated vocabulary; they are returned unchanged otherwise.
    """
    model_config = ConfigDict(extra='forbid')

    consolidated_labels: List[ConsolidatedLabel] = Field(description="Final deduplicated vocabulary labels")
    revised_definition: str = Field(
        description="Contextual factor definition, possibly unchanged. Revise only if the consolidated vocabulary makes the original inaccurate or ambiguous."
    )
    revised_guidelines: List[str] = Field(
        description="Labeling guidelines, possibly unchanged. Update only the items affected by vocabulary changes; otherwise keep them verbatim."
    )
    consolidation_notes: str = Field(
        description="Brief summary of vocabulary consolidation decisions; mention any definition/guideline edit only if it was forced by a vocabulary change."
    )


# Back-compat alias (older name referred only to vocabulary).
VocabularyConsolidationResult = ContextualFactorConsolidationResult
