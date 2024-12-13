from pydantic import BaseModel, Field, create_model
from typing import Literal

class StepSchema(BaseModel):
    explanation: str = Field(..., title="The explanation or reasoning behind the step")
    title: str = Field(..., title="An informative title which is high-level and as abstracted as possible, avoiding specific tools, materials/ingredients, methods, or outcomes.")
    description: str = Field(..., title="A step description/definition that describes the information it should cover, potentially in different tutorial videos.")

class TimedStepSchema(StepSchema):
    start_index: int = Field(..., title="The start index of the step in the transcript")
    end_index: int = Field(..., title="The end index of the step in the transcript")

class StepsSchema(BaseModel):
    steps: list[StepSchema] = Field(..., title="A list of steps that guide novices through the task")

class TimedStepsSchema(BaseModel):
    steps: list[TimedStepSchema] = Field(..., title="A list of steps based on the narration of the tutorial")