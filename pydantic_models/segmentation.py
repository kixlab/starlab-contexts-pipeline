from pydantic import BaseModel, Field, validator

class Subgoal(BaseModel):
    title: str = Field(..., title="A 2-word name/title of the subgoal")
    definition: str = Field(..., title="The detailed description of the subgoal specifying the information it should cover in different videos.")
    dependencies: list[str] = Field(..., title="The list of subgoals that this subgoal depends on.")

    def get_dict(self):
        return {
            "title": self.title,
            "definition": self.definition,
            "dependencies": self.dependencies,
        }

class TaskGraph(BaseModel):
    subgoals: list[Subgoal] = Field(..., title="The list of definitions of subgoals of the procedure with their dependencies")
    def get_dict(self):
        return [subgoal.get_dict() for subgoal in self.subgoals]
    
class VideoSegment(BaseModel):
    start: float = Field(..., title="The start time of the segment in seconds")
    finish: float = Field(..., title="The finish time of the segment in seconds")
    title: str = Field(..., title="The title of subgoal that the segment belongs")
    text: str = Field(..., title="The segment's combined subtitle texts")
    explanation: str = Field(..., title="The jusitifcation on why this segment is assigned to a particular subgoal")

    def get_dict(self):
        return {
            "start": self.start,
            "finish": self.finish,
            "title": self.title,
            "text": self.text,
            "explanation": self.explanation,
        }
    
class VideoSegmentation(BaseModel):
    segments: list[VideoSegment] = Field(..., title="The list of segments of the video")

    def get_dict(self):
        return [segment.get_dict() for segment in self.segments]
    

class IndexVideoSegment(BaseModel):
    ## todo: may need to ask for content-ids with representative frames here?
    title: str = Field(..., title="The title of subgoal that the segment belongs")
    content_ids: list[str] = Field(..., title="The list of content ids that belong to the segment")
    explanation: str = Field(..., title="The jusitifcation on why this segment is assigned to a `title` subgoal")

    def get_dict(self):
        return {
            "title": self.title,
            "content_ids": self.content_ids,
            "explanation": self.explanation,
        }
    
class IndexVideoSegmentation(BaseModel):
    segments: list[IndexVideoSegment] = Field(..., title="The list of segments of the video")

    def get_dict(self):
        return [segment.get_dict() for segment in self.segments]
    
class SegmentSchema(BaseModel):
    title: str = Field(..., title="The title of the subgoal that the segment belongs")
    quotes: list[str] = Field(..., title="The list of quotes that belong to the segment")
    explanation: str = Field(..., title="The justification of why this segment is assigned to a subgoal under `title`")

    def get_dict(self):
        return {
            "quotes": self.quotes,
            "title": self.title,
            "explanation": self.explanation,
        }
    

class SegmentationSchema(BaseModel):
    segments: list[SegmentSchema] = Field(..., title="The comprehensive list of segments of the video")

    def get_dict(self):
        return [segment.get_dict() for segment in self.segments]

    