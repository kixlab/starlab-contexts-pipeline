from pydantic import BaseModel, Field

from typing import Literal

from enum import Enum

from pydantic_models.organization import SummarizedAlignmentSchema, SummarizedAlignmentSchema2

class Relation(BaseModel):
    reference1: list[str] = Field(..., title="the list of relevant quotes from the first source.")
    reference2: list[str] = Field(..., title="the list of relevant quotes from the second source.")
    description: str = Field(..., title="the concise description of the relation between the two sources.")
    reasoning: str = Field(..., title="the reasoning behind the relation.")
    implication: str = Field(..., title="the potential implications of the relation.")
    helpfulness: str = Field(..., title="the potential helpfulness of the relation to the learners.")

class InformationRelations(BaseModel):
    consistent: list[Relation] = Field(..., title="The list of consistent information between sources.")
    complementary: list[Relation] = Field(..., title="The list of complementary information between sources.")
    contradictory: list[Relation] = Field(..., title="The list of contradictory information between sources.")

class InformationEnum(str, Enum):
    additional_information = "`Additional Information`"
    alternative_method = "`Alternative Method`"
    alternative_setting = "`Alternative Setting`"
    alternative_example = "`Alternative Example`"
    
class AlignmentClassificationSchema(BaseModel):
    classification: InformationEnum = Field(..., title="the classification of the new information into one of (1) additional_information, (2) alternative method, (3) alternative_setting, (4) alternative_example.")
    explanation: str = Field(..., title="the explanation of the classification.")
    provenance: str = Field(..., title="the provenance of the new information. why is the new content present only in the new video?")
    helpfulness: str = Field(..., title="the helpfulness of the new content to learners in improving their understanding.")

class ReferenceAlignment(BaseModel):
    alignment_id: str = Field(..., title="the id of the relevant alignment.")
    description: str = Field(..., title="the concise description of the information alignment.")

class AlignmentHookSchema(BaseModel):
    newref: list[str] = Field(..., title="the list of relevant quotes from the new video.")
    alignments: list[ReferenceAlignment] = Field(..., title="The list of alignments between the new video and the previous videos that pertain to `newref`.")
    title: str = Field(..., title="the hook that describes the alignments and can be used to interest learners in the new content or overall comparison.")
    
class AlignmentHooksSchema(BaseModel):
    hooks: list[AlignmentHookSchema] = Field(..., title="The comprehensive and exhaustive list of hooks cover all information alignments")

## V2
    
class InformationAlignmentSchema(SummarizedAlignmentSchema):
    quotes: list[str] = Field(..., title="(must be nonempty) the list of relevant quotes from the current video.")
    other_quotes: list[str] = Field(..., title="(if any) the list of relevant quotes from the previous video.")
    # different_aspects: list[Literal["subgoal", "context", "outcome", "materials", "instructions", "rationale", "tips"]] = Field(..., title="a list of different aspects that the new content pertains to.")

class AlignmentsSchema(BaseModel):
    alignments: list[InformationAlignmentSchema] = Field(..., title="The comprehensive and exhaustive list of new content in the current video")

## V3
class ClassifiedAlignmentSchema(InformationAlignmentSchema): 
    classification: Literal["goal", "materials", "outcome", "instructions", "rationale", "tips", "other"] = Field(..., title="the classification of the new information in terms of its main subject: goal, materials, outcome, instructions, rationale, tips, or other.")

class AlignmentsSchema2(BaseModel):
    supplementary_information: list[ClassifiedAlignmentSchema] = Field(..., title="The list of `new` contents in the video that can be considered `supplementary` to the previous video (i.e., does not contradict or replace any information in the previous video and adds or extends the information in the previous video).")
    contradictory_information: list[ClassifiedAlignmentSchema] = Field(..., title="The list of `new` contents in the video that can be considered `contradictory` to the previous video (i.e., contradicts or replaces any information in the previous video, but presents a different context or approach).")
    more_information_exist: bool = Field(..., title="whether more information exists that was not yet covered in the list so far.")

## V3.1
class ClassifiedAlignmentSchema3(InformationAlignmentSchema): 
    classification: Literal["goal", "objects", "outcome", "instructions", "rationale", "tips", "other"] = Field(..., title="the classification of the new information in terms of its main subject: goal, objects, outcome, instructions, rationale, tips, or other.")

class AlignmentsSchema3(BaseModel):
    supplementary_information: list[ClassifiedAlignmentSchema3] = Field(..., title="The list of `new` contents in the video that can be considered `supplementary` to the previous video (i.e., does not contradict or replace any information in the previous video and adds or extends the information in the previous video).")
    contradictory_information: list[ClassifiedAlignmentSchema3] = Field(..., title="The list of `new` contents in the video that can be considered `contradictory` to the previous video (i.e., contradicts or replaces any information in the previous video, but presents a different context or approach).")


## V4
class AlignmentSchema4(SummarizedAlignmentSchema2): 
    # "materials", "outcome", "setting", "instructions", "explanation", "tips", "tools"
    aspect: Literal["materials", "outcome", "setting", "instructions", "explanation", "tips", "tools", "other"] = Field(..., title="the procedural aspect of the new content in the video: materials, outcome, setting, instructions, explanation, tips, tools, or other.")
    relation: Literal["additional", "alternative"] = Field(..., title="the relation of the new content to other video: additional or alternative. additional means the new content is supplementary to the other video, while alternative means that new content contradicts or is different compare to the other video.")
    importance: int = Field(..., title="the score of the new content in terms of its importance for successful completion of the task. Give a score from 1 to 5, where 1 is the least important (i.e., a minor detail) and 5 is the most important (i.e., a critical step).")
    # supplementary/alternative of each type is...?

class AlignmentsSchema4(BaseModel):
    new_contents_in_1: list[AlignmentSchema4] = Field(..., title="the list of new procedural contents in the video 1 that are not present in the video 2.")
    new_contents_in_2: list[AlignmentSchema4] = Field(..., title="the list of new procedural contents in the video 2 that are not present in the video 1.")