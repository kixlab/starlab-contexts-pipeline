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



### codebook

class LabelExampleSchema(BaseModel):
    content: str = Field(..., title="The example content that would be categorized as the label.")
    context: str = Field(..., title="The context of the example content. The context should include some text (around 10-20 words) surrounding the content and the content itself.")

class LabelSchema(BaseModel):
    id: str = Field(..., title="The id.")
    title: str = Field(..., title="The title.")
    definition: str = Field(..., title="The definition.")
    examples: list[LabelExampleSchema] = Field(..., title="The list of examples.")

class LabelListSchema(BaseModel):
    labels: list[LabelSchema] = Field(..., title="The list of labels.")


### labeled pieces
class LabeledPieceSchema(BaseModel):
    piece_id: int = Field(..., title="The provided id of the piece.")
    label_id: str = Field(..., title="The id of the label.")
    label: str = Field(..., title="The label of the piece.")

class LabeledPiecesSchema(BaseModel):
    labeled_pieces: list[LabeledPieceSchema] = Field(..., title="The list of labeled pieces.")
    

### facet candidates
class FacetValueSchema(BaseModel):
    title: str = Field(..., title="The title of the facet value.")
    definition: str = Field(..., title="The definition of the facet value.")

class CandidateApplicabilityFacetSchema(BaseModel):
    type: Literal["why", "when", "where"] = Field(..., title="The type of the applicability facet.")
    id: str = Field(..., title="The id of the applicability facet.")
    title: str = Field(..., title="The title of the applicability facet.")
    title_plural: str = Field(..., title="The plural of the applicability facet.")
    definition: str = Field(..., title="The definition of the applicability facet.")
    guidelines: list[str] = Field(..., title="The guidelines for the LLM to generate different facet values of the applicability facet.")
    examples: list[FacetValueSchema] = Field(..., title="The examples of the applicability facet.")

class CandidateApplicabilityFacetsSchema(BaseModel):
    candidates: list[CandidateApplicabilityFacetSchema] = Field(..., title="The list of candidate applicability facets.")