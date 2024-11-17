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

# class AggStepMappingSchema(BaseModel):
#     original_step: str = Field(..., title="The original step from one of the lists.")
#     agg_step: str = Field(..., title="The aggregated step that the original step is mapped to.")

class AggStepSchema(BaseModel):
    agg_step: str = Field(..., title="The aggregated step.")
    original_steps_1: list[int] = Field(..., title="The list of ids of original steps from the video 1 that are mapped to the aggregated step.")
    original_steps_2: list[int] = Field(..., title="The list of ids of original steps from the video 2 that are mapped to the aggregated step.")

class AggStepsSchema(BaseModel):
    agg_steps: list[AggStepSchema] = Field(..., title="The list of aggregated steps to achieve the task.")

    # agg_steps: list[str] = Field(..., title="The list of aggregated steps to achieve the task.")
    # assignments_1: list[AggStepMappingSchema] = Field(..., title="The mapping of original steps in the first list to aggregated steps.")
    # assignments_2: list[AggStepMappingSchema] = Field(..., title="The mapping of original steps in the second list to aggregated steps.")


class AggSubgoalSchema(BaseModel):
    title: str = Field(..., title="A 1 to 3 words title of the subgoal")
    description: str = Field(..., title="The description of the subgoal that specifies the information it should cover in tutorial videos")
    original_steps: list[int] = Field(..., title="The list of ids of original steps that are mapped to the subgoal.")

# class AggSubgoalMappingSchema(BaseModel):
#     step: str = Field(..., title="The step that the subgoal is mapped to.")
#     subgoal: str = Field(..., title="The subgoal that the step is mapped to.")

class AggSubgoalsSchema(BaseModel):
    subgoals: list[AggSubgoalSchema] = Field(..., title="The list of subgoals with their steps")
    # assignments: list[AggSubgoalMappingSchema] = Field(..., title="The mapping of steps to subgoals")

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

## V5

class SubgoalSchema(BaseModel):
    explanation: str = Field(..., title="The explanation or reasoning behind the subgoal")
    title: str = Field(..., title="A 1 to 3 words title of the subgoal, which is high-level and as abstracted as possible, avoiding specific tools, materials/ingredients, methods, or outcomes.")
    description: str = Field(..., title="A subgoal description/definition that specifies the `procedural information` it should cover, potentially in different tutorial videos, in terms of the steps, materials, tools, outcomes, and other relevant information.")

class SubgoalSegmentSchema(SubgoalSchema):
    start_index: int = Field(..., title="The start index of the subgoal segment in the transcript")
    end_index: int = Field(..., title="The end index of the subgoal segment in the transcript")

class AggregatedSubgoalSchema(SubgoalSchema):
    original_subgoal_ids: list[int] = Field(..., title="The list of ids of original subgoals that are mapped to the aggregated subgoal.")

class SubgoalsSchema(BaseModel):
    explanation: str = Field(..., title="The explanation or reasoning behind the subgoals")
    subgoals: list[SubgoalSchema] = Field(..., title="The list of subgoals to achieve the task.")

class SubgoalSegmentationSchema(BaseModel):
    explanation: str = Field(..., title="the explanation or reasoning behind the subgoal segmentation")
    segments: list[SubgoalSegmentSchema] = Field(..., title="The comprehensive list of subgoal-based segments in the video.")

class AggregatedSubgoalsSchema(BaseModel):
    explanation: str = Field(..., title="The explanation or reasoning behind the aggregated subgoals")
    subgoals: list[AggregatedSubgoalSchema] = Field(..., title="The list of aggregated subgoals with their original subgoals.")