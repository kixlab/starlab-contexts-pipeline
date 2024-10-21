from pydantic import BaseModel, Field, create_model
from typing import Literal

class Subgoal(BaseModel):
    title: str = Field(..., title="A 2-word name/title of the subgoal")
    definition: str = Field(..., title="The detailed description of the subgoal specifying the information it should cover in different videos.")
    dependencies: list[str] = Field(..., title="The list of subgoals that this subgoal depends on.")

class TaskGraph(BaseModel):
    subgoals: list[Subgoal] = Field(..., title="The list of definitions of subgoals of the procedure with their dependencies")
    
class VideoSegment(BaseModel):
    start: float = Field(..., title="The start time of the segment in seconds")
    finish: float = Field(..., title="The finish time of the segment in seconds")
    title: str = Field(..., title="The title of subgoal that the segment belongs")
    text: str = Field(..., title="The segment's combined subtitle texts")
    explanation: str = Field(..., title="The jusitifcation on why this segment is assigned to a particular subgoal")
    
class VideoSegmentation(BaseModel):
    segments: list[VideoSegment] = Field(..., title="The list of segments of the video")
    

class IndexVideoSegment(BaseModel):
    title: str = Field(..., title="The title of subgoal that the segment belongs")
    content_ids: list[str] = Field(..., title="The list of content ids that belong to the segment")
    explanation: str = Field(..., title="The jusitifcation on why this segment is assigned to a `title` subgoal")
    
class IndexVideoSegmentation(BaseModel):
    segments: list[IndexVideoSegment] = Field(..., title="The list of segments of the video")

class BaseSegmentSchema(BaseModel):
    quotes: list[str] = Field(..., title="The list of quotes that belong to the segment")
    explanation: str = Field(..., title="The justification of why this segment is assigned to a subgoal under `title`")
    

def get_segmentation_schema(titles: list[str]):
    TitleLiteral = Literal[tuple(titles)]
    SegmentSchema = create_model(
        'SegmentSchema',
        title=(TitleLiteral, Field(..., title="The title of the subgoal that the segment belongs")),
        __base__=BaseSegmentSchema,
    )


    SegmentationSchema = create_model(
        'SegmentationSchema',
        segments=(list[SegmentSchema], Field(..., title="The list of segments of the video")),
        __base__=BaseModel,
    )

    return SegmentationSchema

class SubgoalSchema(BaseModel):
    title: str = Field(..., title="A short title of the subgoal")
    description: str = Field(..., title="The description of the subgoal.")
    outcome: str = Field(..., title="The expected outcome of the subgoal")

class ListSubgoalsSchema(BaseModel):
    subgoals: list[SubgoalSchema] = Field(..., title="The list of subgoals with their descriptions and expected outcomes")

class ProceduralInformationSchema(BaseModel):
    information: str = Field(..., title="The piece of procedural information.")
    type: Literal["goal", "materials", "outcome", "instructions", "rationale", "tips", "tools", "other"] = Field(..., title="The type of the procedural information")
    reasoning: str = Field(..., title="The reasoning behind importance of the information.")
    importance: Literal["essential", "optional", "irrelevant"] = Field(..., title="The importance of the information for completing the task")

class AllProceduralInformationSchema(BaseModel):
    all: list[ProceduralInformationSchema] = Field(..., title="The list of procedural information included in the provided contents. Empty list if no procedural information is present.")

    
    
