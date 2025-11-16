from pydantic import BaseModel, Field, ConfigDict
from typing import Literal

### information units
class BaseInfoSchema(BaseModel):    
    model_config = ConfigDict(extra='forbid')
    content: str = Field(..., title="The text of the information piece.")
    type: Literal["Greeting", "Overview", "Method", "Supplementary", "Explanation", "Description", "Conclusion", "Miscellaneous"] = Field(..., title="The type of the information piece: Greeting|Overview|Method|Supplementary|Explanation|Description|Conclusion|Miscellaneous")
    subtype: Literal["Opening", "Closing", "Goal", "Motivation", "Briefing", "Subgoal", "Instruction", "Tool", "Tip", "Warning", "Justification", "Effect", "Status", "Context", "Tool Specification", "Outcome", "Reflection", "Side Note", "Self-promotion", "Bridge", "Filler"] = Field(..., title="The subtype of the information piece: Opening|Closing|Goal|Motivation|Briefing|Subgoal|Instruction|Tool|Tip|Warning|Justification|Effect|Status|Context|Tool Specification|Outcome|Reflection|Side Note|Self-promotion|Bridge|Filler")
    start: float = Field(..., title="The start time of the information piece. `null` if no timing metadata exists.")
    end: float = Field(..., title="The end time of the information piece. `null` if no timing metadata exists.")


class InformationPiecesSchema(BaseModel):
    model_config = ConfigDict(extra='forbid')
    pieces: list[BaseInfoSchema] = Field(..., title="The list of information pieces.")



### segment labels

# class LabelExampleSchema(BaseModel):
#     content: str = Field(..., title="The content that would be labeled as the segment label.")
#     context: str = Field(..., title="The context of the content. The context should include some text (around 10-20 words) surrounding the content as well as the content itself.")

class LabelSchema(BaseModel):
    model_config = ConfigDict(extra='forbid')
    id: str = Field(..., title="The id of the segment label.")
    label: str = Field(..., title="The text representing the segment label. Less than 2-3 words.")
    definition: str = Field(..., title="The definition of the segment label.")
    # examples: list[LabelExampleSchema] = Field(..., title="1-2 short representative content and context that would be labeled as the segment label.")

class VocabularySchema(BaseModel):
    model_config = ConfigDict(extra='forbid')
    vocabulary: list[LabelSchema] = Field(..., title="The list of canonical segment labels (i.e., vocabulary for temporal segmentation) used for temporal segmentation along a task context aspect.")

class SegmentationFacetSchema(BaseModel):
    model_config = ConfigDict(extra='forbid')
    aspect: str = Field(..., title="The title of the aspect. Less than 2-3 words.")
    aspect_plural: str = Field(..., title="The plural form of the aspect.")
    type: Literal["when", "why", "where", "what", "how"] = Field(..., 
    title="The type of the aspect of a task context.")
    definition: str = Field(..., title="The definition of the aspect (i.e., segmentation): what the aspect is about, what it means, etc.")
    guidelines: list[str] = Field(..., title="The guidelines for the LLM to temporally segment the tutorial-style transcript along this aspect.")
    segment_labels: list[LabelSchema] = Field(..., title="The list of segment labels used for temporal segmentation along this aspect.")

### labeled pieces
class LabeledPieceSchema(BaseModel):
    model_config = ConfigDict(extra='forbid')
    piece_id: int = Field(..., title="The provided id of the piece.")
    label_id: str = Field(..., title="The id of the segment label.")
    label: str = Field(..., title="The segment label (without the id).")

class LabeledPiecesSchema(BaseModel):
    model_config = ConfigDict(extra='forbid')
    labeled_pieces: list[LabeledPieceSchema] = Field(..., title="The list of pieces with corresponding segment labels.")

### facet candidates
# class FacetValueSchema(BaseModel):
#     model_config = ConfigDict(extra='forbid')
#     label: str = Field(..., title="An example segment label. Less than 2-3 words.")
#     definition: str = Field(..., title="The definition of the example segment label.")

class CandidateSegmentationFacetSchema(BaseModel):
    model_config = ConfigDict(extra='forbid')
    id: str = Field(..., title="The id of the aspect of a task context.")
    aspect: str = Field(..., title="The title of the aspect. Less than 2-3 words.")
    aspect_plural: str = Field(..., title="The plural form of the aspect.")
    type: Literal["when", "why", "where", "what", "how"] = Field(..., 
    title="The type of the aspect of a task context.")
    justification: str = Field(..., title="A brief justification of the choice of the aspect and the type of segmentation")
    definition: str = Field(..., title="The definition of the aspect (i.e., segmentation): what the aspect is about, what it means, etc.")
    guidelines: list[str] = Field(..., title="The guidelines for the LLM to temporally segment the tutorial-style transcript along this aspect.")
    segment_labels: list[LabelSchema] = Field(..., title="The full canonical vocabulary for temporal segmentation along this aspect.")
    segmentations: list[LabeledPiecesSchema] = Field(..., title="The list of segmentations of the provided tutorials along this aspect with the corresponding segment labels.")


class CandidateSegmentationFacetsSchema(BaseModel):
    model_config = ConfigDict(extra='forbid')
    candidates: list[CandidateSegmentationFacetSchema] = Field(..., title="The list of candidate task context aspects.")