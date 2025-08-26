from pydantic import BaseModel, Field


class InstructionSchema(BaseModel):
    instruction: str = Field(..., title="The fine-grained instruction for the user to follow.")
    details: list[str] = Field(..., title="The list of details relevant to the instruction (e.g., explanations, tips, descriptions).")
    

class AllInstructionsHSchema(BaseModel):
    instructions: list[InstructionSchema] = Field(..., title="The list of fine-grained instructions in the order shown in the tutorial.")

class AllInstructionsSchema(BaseModel):
    instructions: list[str] = Field(..., title="The list of fine-grained instructions in the order shown in the tutorial.")