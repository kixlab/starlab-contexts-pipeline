from pydantic import BaseModel, Field, create_model
from typing import Literal, Tuple

class InstructionSchema(BaseModel):
    instruction: str = Field(..., title="The instruction for the task extracted from the subtitle")
    expected_outcome: str = Field(..., title="The expected outcome/effect/result of the instruction")

class TimedInstructionSchema(InstructionSchema):
    # index_range: Tuple[int, int] = Field(..., title="The start and end indices of the subtitle in the transcript")
    start: int = Field(..., title="The start index of the subtitle in the transcript")
    end: int = Field(..., title="The end index of the subtitle in the transcript")

class TimedInstructionsSchema(BaseModel):
    instructions: list[TimedInstructionSchema] = Field(..., title="A list of instructions based on the subtitles of the tutorial")