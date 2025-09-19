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

class AnswerExampleSchema(BaseModel):
    content: str = Field(..., title="The content that would lead to the answer.")
    context: str = Field(..., title="The context of the content. The context should include some text (around 10-20 words) surrounding the content as well as the content itself.")

class AnswerSchema(BaseModel):
    id: str = Field(..., title="The id of the answer.")
    answer: str = Field(..., title="The answer to the question. Less than 2-3 words.")
    definition: str = Field(..., title="The elaboration of what the answer means.")
    examples: list[AnswerExampleSchema] = Field(..., title="1-2 contents that would lead to the answer.")

class AnswerListSchema(BaseModel):
    answers: list[AnswerSchema] = Field(..., title="The list of canonical answers to the question.")


### labeled pieces
class LabeledPieceSchema(BaseModel):
    piece_id: int = Field(..., title="The provided id of the piece.")
    answer_id: str = Field(..., title="The id of the answer.")
    answer: str = Field(..., title="The answer to the question.")

class LabeledPiecesSchema(BaseModel):
    labeled_pieces: list[LabeledPieceSchema] = Field(..., title="The list of pieces with corresponding answers.")

### facet candidates
class FacetValueSchema(BaseModel):
    answer: str = Field(..., title="An example answer to the question-facet. Less than 2 words.")
    definition: str = Field(..., title="The elaboration of what the answer means.")

class CandidateFacetSchema(BaseModel):
    id: str = Field(..., title="The id of the facet.")
    type: Literal["why", "when", "where"] = Field(..., title="The type of the applicability facet.")
    title: str = Field(..., title="The title of the facet. Less than 2-3 words.")
    title_plural: str = Field(..., title="The plural of the facet.")
    question: str = Field(..., title="The question that represents the facet. Less than 20 words")
    # answer_format_guidelines: list[str] = Field(..., title="The format guidelines for the LLM when answering the question (i.e., extracting the facet values).")
    answer_guidelines: list[str] = Field(..., title="The guidelines for the LLM to extract the facet value (i.e., answer) from the tutorial-style transcript.")
    examples: list[FacetValueSchema] = Field(..., title="1-2 short example answers to the question. Less than 2 words each.")

class CandidateFacetsSchema(BaseModel):
    candidates: list[CandidateFacetSchema] = Field(..., title="The list of candidate applicability facets.")