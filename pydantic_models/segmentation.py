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
    ## todo: may need to ask for content-ids with representative frames here?
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