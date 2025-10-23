from pydantic import BaseModel, Field
from typing import Literal

class SegmentSchema(BaseModel):
    text: str = Field(..., title="The text of the information piece.")
    type: str = Field(..., title="The type of the information piece according to the taxonomy.")

class SegmentationSchema(BaseModel):
    segments: list[SegmentSchema] = Field(..., title="The list of information pieces.")

class ClassificationSchema(BaseModel):
    category: str = Field(..., title="The category of the sentence.")




### information units
class BaseInfoSchema(BaseModel):    
    content: str = Field(..., title="The text of the information piece.")
    content_type: Literal["Greeting", "Overview", "Method", "Supplementary", "Explanation", "Description", "Conclusion", "Miscellaneous"] = Field(..., title="The type of the information piece: Greeting|Overview|Method|Supplementary|Explanation|Description|Conclusion|Miscellaneous")
    start: float = Field(..., title="The start time of the information piece. `null` if no timing metadata exists.")
    end: float = Field(..., title="The end time of the information piece. `null` if no timing metadata exists.")


class InformationPiecesSchema(BaseModel):
    pieces: list[BaseInfoSchema] = Field(..., title="The list of information pieces classified into method, description, explanation, supplementary, or other.")



### segment labels

class LabelExampleSchema(BaseModel):
    content: str = Field(..., title="The content that would be labeled as the segment label.")
    context: str = Field(..., title="The context of the content. The context should include some text (around 10-20 words) surrounding the content as well as the content itself.")

class LabelSchema(BaseModel):
    id: str = Field(..., title="The id of the segment label.")
    label: str = Field(..., title="Text representing the segment label. Less than 2-3 words.")
    definition: str = Field(..., title="The elaboration of what the segment label means.")
    examples: list[LabelExampleSchema] = Field(..., title="1-2 short representative content and context that would be labeled as the segment label.")

class VocabularySchema(BaseModel):
    vocabulary: list[LabelSchema] = Field(..., title="The list of canonical segment labels (i.e., vocabulary for temporal segmentation).")


### labeled pieces
class LabeledPieceSchema(BaseModel):
    piece_id: int = Field(..., title="The provided id of the piece.")
    label_id: str = Field(..., title="The id of the segment label.")
    label: str = Field(..., title="The segment label (without the id).")

class LabeledPiecesSchema(BaseModel):
    labeled_pieces: list[LabeledPieceSchema] = Field(..., title="The list of pieces with corresponding segment labels.")

### facet candidates
class FacetValueSchema(BaseModel):
    label: str = Field(..., title="An example segment label. Less than 2-3 words.")
    definition: str = Field(..., title="The elaboration of what the segment label means.")

class CandidateSegmentationFacetSchema(BaseModel):
    id: str = Field(..., title="The id of the aspect of a task context.")
    aspect: str = Field(..., title="The title of the aspect. Less than 2-3 words.")
    aspect_plural: str = Field(..., title="The plural form of the aspect.")
    type: Literal["when", "why", "where", "what", "how"] = Field(..., 
    title="The type of the aspect of a task context.")
    justification: str = Field(..., title="A brief justification of the choice of the aspect and the type of segmentation")
    segmentation: str = Field(..., title="The description of how the task can be segmented along this aspect.")
    segmentation_guidelines: list[str] = Field(..., title="The guidelines for the LLM to extract the segment labels from the tutorial-style transcript.")
    segment_labels: list[FacetValueSchema] = Field(..., title="The list of segment labels.")

class CandidateSegmentationFacetsSchema(BaseModel):
    candidates: list[CandidateSegmentationFacetSchema] = Field(..., title="The list of candidate aspects of a task context.")