from pydantic import BaseModel, Field, create_model
from typing import Literal

class TranscriptMappingSchema(BaseModel):
    index: int = Field(..., title="The index of the sentence in the narration.")
    steps: list[str] = Field(..., title="The list of steps that the sentence is mapped to. If the sentence is not mapped to any step, the list is empty.")
    relevance: Literal["essential", "optional", "irrelevant"] = Field(..., title="The relevance of the sentence to the task.")

class TranscriptAssignmentsSchema(BaseModel):
    assignments: list[TranscriptMappingSchema] = Field(..., title="The mapping of sentences in the narration to steps.")

class StepsSchema(BaseModel):
    steps: list[str] = Field(..., title="A comprehensive list of steps to achieve the task.")

class AggStepMappingSchema(BaseModel):
    original_step: str = Field(..., title="The original step from one of the lists.")
    agg_step: str = Field(..., title="The aggregated step that the original step is mapped to.")

class AggStepsSchema(BaseModel):
    agg_steps: list[str] = Field(..., title="The list of aggregated steps to achieve the task.")
    assignments_1: list[AggStepMappingSchema] = Field(..., title="The mapping of original steps in the first list to aggregated steps.")
    assignments_2: list[AggStepMappingSchema] = Field(..., title="The mapping of original steps in the second list to aggregated steps.")


class AggSubgoalMappingSchema(BaseModel):
    step: str = Field(..., title="The step that the subgoal is mapped to.")
    subgoal: str = Field(..., title="The subgoal that the step is mapped to.")

class AggSubgoalSchema(BaseModel):
    title: str = Field(..., title="A 1 to 3 words title of the subgoal")
    description: str = Field(..., title="The description of the subgoal that specifies the information it should cover in tutorial videos")

class AggSubgoalsSchema(BaseModel):
    subgoals: list[AggSubgoalSchema] = Field(..., title="The list of subgoals with their steps")
    assignments: list[AggSubgoalMappingSchema] = Field(..., title="The mapping of steps to subgoals")

def get_segmentation_schema_v4(titles):
    if not titles:
        TitleLiteral = str
    else:
        TitleLiteral = Literal[tuple(titles)]
    SegmentSchema = create_model(
        'SegmentSchema',
        step=(TitleLiteral, Field(..., title="The step that the segment belongs to")),
        start_index=(int, Field(..., title="The start index of the segment in the transcript")),
        end_index=(int, Field(..., title="The end index of the segment in the transcript")),
        __base__=BaseModel,
    )


    SegmentationSchema = create_model(
        'SegmentationSchema',
        segments=(list[SegmentSchema], Field(..., title="The comprehensive list of segments of the video")),
        __base__=BaseModel,
    )

    return SegmentationSchema

    
    
