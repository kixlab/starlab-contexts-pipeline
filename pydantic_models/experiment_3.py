from pydantic import BaseModel, Field
from typing import Literal

class SegmentSchema(BaseModel):
    text: str = Field(..., title="The text of the information piece.")
    type: str = Field(..., title="The type of the information piece according to the taxonomy.")

class SegmentationSchema(BaseModel):
    segments: list[SegmentSchema] = Field(..., title="The list of information pieces.")

class ClassificationSchema(BaseModel):
    category: str = Field(..., title="The category of the sentence.")





class BaseInfoSchema(BaseModel):    
    content: str = Field(..., title="The text of the information piece.")
    content_type: Literal["Greeting", "Overview", "Method", "Supplementary", "Explanation", "Description", "Conclusion", "Miscellaneous"] = Field(..., title="The type of the information piece: Greeting|Overview|Method|Supplementary|Explanation|Description|Conclusion|Miscellaneous")
    start: float = Field(..., title="The start time of the information piece. `null` if no timing metadata exists.")
    end: float = Field(..., title="The end time of the information piece. `null` if no timing metadata exists.")

class InfoSchema(BaseInfoSchema):
    procedure_segments: list[str] = Field(..., title="The list of procedure segments, e.g., ['segment1', 'segment2', ...]")

class InformationPiecesSchema(BaseModel):
  pieces: list[BaseInfoSchema] = Field(..., title="The list of information pieces classified into method, description, explanation, supplementary, or other.")

class ProcedureSegmentsSchema(BaseModel):
  segments: list[str] = Field(..., title="The list of procedure segments extracted from each `method` item, e.g., ['segment1', 'segment2', ...]")

class StructuredPiecesSchema(BaseModel):  
    pieces: list[InfoSchema] = Field(..., title="The list of information pieces with timing metadata.")

class ContextStepSchema(BaseInfoSchema):
    context_step: str = Field(..., title="The context step of the information piece.")

class LabeledPiecesSchema(BaseModel):
    pieces: list[ContextStepSchema] = Field(..., title="The list of information pieces with context steps.")

    
    