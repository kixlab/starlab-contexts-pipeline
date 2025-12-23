from pydantic import BaseModel, Field, ConfigDict
from typing import Literal, Sequence, Type

### information units
class BaseInfoSchema(BaseModel):    
    model_config = ConfigDict(extra='forbid')
    content: str = Field(..., title="The text of the information piece.")
    content_type: Literal["Greeting - Opening", "Greeting - Closing", "Overview - Goal", "Overview - Motivation", "Overview - Briefing", "Method - Subgoal", "Method - Instruction", "Method - Tool", "Supplementary - Tip", "Supplementary - Warning", "Explanation - Justification", "Explanation - Effect", "Description - Status", "Description - Context", "Description - Tool Specification", "Conclusion - Outcome", "Conclusion - Reflection", "Miscellaneous - Side Note", "Miscellaneous - Self-promotion", "Miscellaneous - Bridge", "Miscellaneous - Filler"] = Field(..., title="The type of information piece: Greeting - Opening|Greeting - Closing|Overview - Goal|Overview - Motivation|Overview - Briefing|Method - Subgoal|Method - Instruction|Method - Tool|Supplementary - Tip|Supplementary - Warning|Explanation - Justification|Explanation - Effect|Description - Status|Description - Context|Description - Tool Specification|Conclusion - Outcome|Conclusion - Reflection|Miscellaneous - Side Note|Miscellaneous - Self-promotion|Miscellaneous - Bridge|Miscellaneous - Filler")
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
def _deduplicate_preserving_order(values: Sequence[str]) -> tuple[str, ...]:
    """Return a tuple of unique strings in their original order."""
    return tuple(dict.fromkeys(values))


def _literal_from_choices(choices: Sequence[str]):
    """Build a Literal that only allows the provided string choices."""
    deduped = _deduplicate_preserving_order(choices)
    if not deduped:
        raise ValueError("label choices must contain at least one item")
    return Literal.__getitem__(deduped)


def _schema_name(base: str, token_source: Sequence[str] | int) -> str:
    token = abs(hash(tuple(token_source if isinstance(token_source, Sequence) else (token_source, ))))
    return f"{base}_{token}"


def create_labeled_piece_schema(label_choices: Sequence[str]) -> Type[BaseModel]:
    """
    Dynamically create a LabeledPieceSchema whose `label` field is restricted to
    the provided label_choices.
    """
    label_literal = _literal_from_choices(label_choices)
    annotations = {
        "piece_id": int,
        "label_id": int,
        "label": label_literal,
    }
    namespace = {
        "__annotations__": annotations,
        "model_config": ConfigDict(extra='forbid'),
        "piece_id": Field(..., title="The provided id of the piece."),
        "label_id": Field(..., title="The id of the segment label."),
        "label": Field(..., title="The segment label."),
        "__module__": __name__,
    }
    class_name = _schema_name("LabeledPieceSchema", label_choices)
    return type(class_name, (BaseModel,), namespace)


def create_labeled_pieces_schema(
    *, label_choices: Sequence[str], min_length: int
) -> Type[BaseModel]:
    """
    Dynamically create a LabeledPiecesSchema requiring at least `min_length`
    labeled pieces, each of which uses the provided label choices.
    """
    if min_length < 0:
        raise ValueError("min_length must be >= 0")
    labeled_piece_schema = create_labeled_piece_schema(label_choices)
    annotations = {
        "labeled_pieces": list[labeled_piece_schema],
    }
    namespace = {
        "__annotations__": annotations,
        "model_config": ConfigDict(extra='forbid'),
        "labeled_pieces": Field(
            ...,
            min_length=min_length,
            max_length=min_length,
            title="The list of pieces with corresponding segment labels.",
        ),
        "__module__": __name__,
    }
    class_name = _schema_name(
        "LabeledPiecesSchema", (*label_choices, min_length)
    )
    return type(class_name, (BaseModel,), namespace)

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
    # segmentations: list[LabeledPiecesSchema] = Field(..., title="The list of segmentations of the provided tutorials along this aspect with the corresponding segment labels.")


class CandidateSegmentationFacetsSchema(BaseModel):
    model_config = ConfigDict(extra='forbid')
    candidates: list[CandidateSegmentationFacetSchema] = Field(..., title="The list of candidate task context aspects.")