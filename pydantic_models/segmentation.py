from pydantic import BaseModel, Field, create_model
from typing import Literal

class TranscriptMappingSchema(BaseModel):
    index: int = Field(..., title="The index of the sentence in the narration.")
    steps: list[str] = Field(..., title="The list of steps that the sentence is mapped to. If the sentence is not mapped to any step, the list is empty.")
    relevance: Literal["essential", "optional", "irrelevant"] = Field(..., title="The relevance of the sentence to the task.")

class TranscriptAssignmentsSchema(BaseModel):
    assignments: list[TranscriptMappingSchema] = Field(..., title="The mapping of sentences in the narration to steps.")

class StepInformationSchema(BaseModel):
    informationType: Literal["instruction", "explanation", "tip", "note"] = Field(..., title="The type of the information. It can be instruction (i.e., procedural information that guides the user to perform the step), explanation (i.e., the justifications and reasons presented in the tutorial for performing the step), tip (i.e., the tips presented in the tutorial that can help in performing the step(s) easier, faster, or more efficiently), note (i.e., the additional useful information that is relevant to the step).")
    information: str = Field(..., title="The information about the step that is extracted from the video narration.")
    reference: list[int] = Field(..., title="The list of indices of sentences in the narration that the information is extracted from.")

class ObjectSchema(BaseModel):
    last_step: int = Field(..., title="The index of the last step where the object, entity, or its state appeared as either input or output. If the object is not mentioned in any of the previous steps at all, the value should be 0.")
    name: str = Field(..., title="The name (if applicable state) of the material or tool.")

class StepSchema(BaseModel):
    """
    - `Actions`: A single verb describing how the materials are modified or acted upon.
	- `Materials`: Objects (physical or digital) that are being modified or processed in the current step. Examples include states of physical/digital objects, elements, ingredients, or components.
	- `Tools`: Objects (physical or digital) or methods used to enable or perform the action on the materials. Examples include software features, physical instruments, or other utilities.
	- `Outcomes`: Objects (physical or digital) that result from the action. Outcomes from one step often become the materials for the next step, unless they are final outcomes.
    """
    step: str = Field(..., title="The step that is extracted from the video following the format: [Action] [Materials] with [Tools] to produce [Outcomes].")
    index: int = Field(..., title="The index of the step in the tutorial. Starts from 1.")
    action: str = Field(..., title="A single verb describing how the materials are modified or acted upon.")
    inputs: list[ObjectSchema] = Field(..., title="Materials or tools (in particular state) that are being used in the current step. Split materials and tools one-by-one, so that each element of the list describes a single material or tool.")
    outcomes: list[str] = Field(..., title="Similar to `materials`, results of the action being applied on materials with tools. Split outcomes one-by-one, so that each element of the list describes a single outcome.")

class StepWithInformationSchema(StepSchema):
    information: list[StepInformationSchema] = Field(..., title="The comprehensive list of information about the step found in the video narration. ")

class StepsSchema(BaseModel):
    steps: list[StepSchema] = Field(..., title="A comprehensive list of steps in the tutorial for achieving the task.")

class StepsWithInformationSchema(BaseModel):
    steps: list[StepWithInformationSchema] = Field(..., title="A comprehensive list of steps in the tutorial for achieving the task with their information.")

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
    # description: str = Field(..., title="A subgoal description/definition that specifies the `procedural information` it should cover, potentially in different tutorial videos, in terms of the steps, materials, tools, outcomes, and other relevant information.")
    expected_outcome_description: list[str] = Field(..., title="The list of expected outcomes that the subgoal should cover.")

class SubgoalSegmentSchema(SubgoalSchema):
    start_index: int = Field(..., title="The start index of the subgoal segment in the transcript")
    end_index: int = Field(..., title="The end index of the subgoal segment in the transcript")

class AggregatedSubgoalSchema(SubgoalSchema):
    original_subgoal_ids: list[int] = Field(..., title="The list of ids of original subgoals that are mapped to the aggregated subgoal.")

class SubgoalDescriptionSchema(SubgoalSchema):
    description: str = Field(..., title="A subgoal description following the template: `Apply [Method] using [Materials] [Tools] to achieve [Outcomes] to achieve [Subgoal]`")
    method: str = Field(..., title="The method or technique that is used to achieve the subgoal. Must be 1 to 3 words.")
    materials: list[str] = Field(..., title="The list of materials or ingredients that are used to achieve the subgoal.")
    tools: list[str] = Field(..., title="The list of tools or instruments that are used to achieve the subgoal.")
    outcomes: list[str] = Field(..., title="The list of outcomes that are produced by achieving the subgoal.")

class SubgoalsSchema(BaseModel):
    explanation: str = Field(..., title="The explanation or reasoning behind the subgoals")
    subgoals: list[SubgoalSchema] = Field(..., title="The list of subgoals to achieve the task.")

class SubgoalDescriptionsSchema(BaseModel):
    explanation: str = Field(..., title="The explanation or reasoning behind the subgoal descriptions")
    subgoals: list[SubgoalDescriptionSchema] = Field(..., title="The list of subgoal descriptions to achieve the task.")


class SubgoalSegmentationSchema(BaseModel):
    explanation: str = Field(..., title="the explanation or reasoning behind the subgoal segmentation")
    segments: list[SubgoalSegmentSchema] = Field(..., title="The comprehensive list of subgoal-based segments in the video.")

class AggregatedSubgoalsSchema(BaseModel):
    explanation: str = Field(..., title="The explanation or reasoning behind the aggregated subgoals")
    subgoals: list[AggregatedSubgoalSchema] = Field(..., title="The list of aggregated subgoals with their original subgoals.")