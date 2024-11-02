from pydantic import BaseModel, Field

from typing import Literal

from enum import Enum

from pydantic_models.organization import SummarizedAlignmentSchema2


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