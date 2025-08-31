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

class ItemExampleSchema(BaseModel):
    context: str = Field(..., title="The context of the example. The context should include some text (around 10-20 words) surrounding the content and the content itself.")
    content: str = Field(..., title="The content of the example.")

class ItemSchema(BaseModel):
    id: str = Field(..., title="The id of the item.")
    title: str = Field(..., title="The title")
    definition: str = Field(..., title="The definition")
    examples: list[ItemExampleSchema] = Field(..., title="The list of examples. There should be at least 1 example and no more than 3 examples.")

class ItemListSchema(BaseModel):
    items: list[ItemSchema] = Field(..., title="The list of items.")


### labeled pieces
class LabeledPieceSchema(BaseModel):
    piece_id: int = Field(..., title="The provided id of the piece.")
    label_id: str = Field(..., title="The id of the label.")
    label: str = Field(..., title="The label of the piece.")

class LabeledPiecesSchema(BaseModel):
    labeled_pieces: list[LabeledPieceSchema] = Field(..., title="The list of labeled pieces.")
    