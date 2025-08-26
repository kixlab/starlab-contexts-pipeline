from pydantic import BaseModel, Field

class SubgoalSchema(BaseModel):
    title: str = Field(..., title="A 1 to 3 words title of the subgoal.")
    description: str = Field(..., title="A subgoal description/definition.")

class SubgoalSegmentSchema(SubgoalSchema):
    start: int = Field(..., title="The start index of the subgoal segment in the transcript")
    end: int = Field(..., title="The end index of the subgoal segment in the transcript")

class AggregatedSubgoalSchema(SubgoalSchema):
    original_subgoal_ids: list[int] = Field(..., title="The list of ids of original subgoals that are mapped to the aggregated subgoal.")

class SubgoalsSchema(BaseModel):
    subgoals: list[SubgoalSchema] = Field(..., title="The comprehensive list of subgoals.")