from pydantic import BaseModel, Field


class StepSchema(BaseModel):
    step: str = Field(..., title="The step for the user to follow.")
    description: str = Field(..., title="The description of the step.")

class StepUpdatedSchema(StepSchema):
    updated: str = Field(..., title="If changed, write `changed`, if added newly, write `added`, if not changed, write `not changed`.")

class StepTaxonomySchema(BaseModel):
    steps: list[StepSchema] = Field(..., title="The list of steps in the taxonomy.")

class StepTaxonomyUpdateSchema(BaseModel):
    steps: list[StepUpdatedSchema] = Field(..., title="The list of steps in the taxonomy.")


class LabeledInfoSchema(BaseModel):
    info: str = Field(..., title="A piece of information relevant to the task.")
    step_label: str = Field(..., title="The label of the step that this piece of information belongs to.")

class LabeledInfoListSchema(BaseModel):
    infos: list[LabeledInfoSchema] = Field(..., title="The list of pieces of information and their labels.")
    updated_taxonomy: list[StepUpdatedSchema] = Field(..., title="The updated taxonomy with the new steps if any.")
