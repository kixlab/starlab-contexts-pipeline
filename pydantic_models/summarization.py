from pydantic import BaseModel, Field

class StepSummarySchema(BaseModel):
    # "materials", "outcome", "setting", "instructions", "explanation", "tips", "tools"

    # "materials", "outcome", "setting", "instructions", "explanation", "tips", "tools"

    ### context
    # subgoal: str = Field(..., title="The detailed description of the subgoal/stage based on the narration.")
    # subgoal_content_ids: list[int] = Field(..., title="A list of narration ids that mention the subgoal.")
    materials: list[str] = Field(..., title="A comprehensive list of materials or ingredients used in the tutorial for the step(s) along with their visual descriptions.")
    materials_content_ids: list[int] = Field(..., title="A list of narration ids that mention the materials.")
    outcomes: list[str] = Field(..., title="A comprehensive list of the outcomes or results of the step(s) along with their visual descriptions.")
    outcomes_content_ids: list[int] = Field(..., title="A list of narration ids that mention the outcome.")
    tools: list[str] = Field(..., title="A comprehensive list of tools or equipments used in the tutorial for the step(s) along with their visual descriptions.")
    tools_content_ids: list[int] = Field(..., title="A list of narration ids that mention the tools.")
    # setting: str = Field(..., title="The context in which the step is performed, including the location, time, and other relevant details.")
    # setting_content_ids: list[int] = Field(..., title="A list of narration ids that mention the setting.")
    
    ### method
    instructions: str = Field(..., title="The instructions presented in the tutorial for the step(s).")
    instructions_content_ids: list[int] = Field(..., title="A list of narration ids that mention the instructions.")
    
    explanations: str = Field(..., title="The justifications and reasons presented in the tutorial for performing the step.")
    explanations_content_ids: list[int] = Field(..., title="A list of narration ids that mention the explanations.")
    
    tips: str = Field(..., title="The tips presented in the tutorial that can help in performing the step(s) easier, faster, or more efficiently.")
    tips_content_ids: list[int] = Field(..., title="A list of narration ids that mention the tips.")
    warnings: str = Field(..., title="The warnings presented in the tutorial that can help avoid mistakes when performing the step(s).")
    warnings_content_ids: list[int] = Field(..., title="A list of narration ids that mention the warnings.")

class SubgoalSummarySchema(BaseModel):
    # "materials", "outcome", "setting", "instructions", "explanation", "tips", "tools"

    ### context
    # subgoal: str = Field(..., title="The detailed description of the subgoal/stage based on the narration.")
    # subgoal_content_ids: list[int] = Field(..., title="A list of narration ids that mention the subgoal.")
    materials: list[str] = Field(..., title="A comprehensive list of materials or ingredients used in the tutorial to complete the subgoal/stage along with their visual descriptions.")
    materials_content_ids: list[int] = Field(..., title="A list of narration ids that mention the materials.")
    outcomes: list[str] = Field(..., title="A comprehensive list of the outcomes or results of completing the subgoal/stage along with their visual descriptions.")
    outcomes_content_ids: list[int] = Field(..., title="A list of narration ids that mention the outcome.")
    tools: list[str] = Field(..., title="A comprehensive list of tools or equipments used in the tutorial to complete the the subgoal/stage along with their visual descriptions.")
    tools_content_ids: list[int] = Field(..., title="A list of narration ids that mention the tools.")
    # setting: str = Field(..., title="The context in which the step is performed, including the location, time, and other relevant details.")
    # setting_content_ids: list[int] = Field(..., title="A list of narration ids that mention the setting.")
    
    ### method
    instructions: str = Field(..., title="The instructions presented in the tutorial for completing the subgoal/stage.")
    instructions_content_ids: list[int] = Field(..., title="A list of narration ids that mention the instructions.")
    
    explanations: str = Field(..., title="The justifications and reasons presented in the tutorial for performing the steps/instructions for completing the subgoal/stage.")
    explanations_content_ids: list[int] = Field(..., title="A list of narration ids that mention the explanations.")
    
    tips: str = Field(..., title="The tips presented in the tutorial that can help in completing the subgoal/stage easier, faster, or more efficiently.")
    tips_content_ids: list[int] = Field(..., title="A list of narration ids that mention the tips.")
    warnings: str = Field(..., title="The warnings presented in the tutorial that can help avoid mistakes when completing the subgoal/stage.")
    warnings_content_ids: list[int] = Field(..., title="A list of narration ids that mention the warnings.")


class ObjectSchema(BaseModel):
    object_name: str = Field(..., title="The name of the object.")
    caption: str = Field(..., title="The visual description of the object that starts with `A photo of a `.")
    description: str = Field(..., title="The detailed description of the object based on the narration.")
    content_ids: list[int] = Field(..., title="A list of narration ids that describe the object.")

class StepSchema(BaseModel):
    explanation: str = Field(..., title="The reasoning on why this step is considered a single action and split out from other steps.")
    description: str = Field(..., title="The description of the step that specifies the single action to be performed, the tools used, and the materials required.")
    
    instructions: str = Field(..., title="The instructions presented in the tutorial for completing the step.")
    instructions_content_ids: list[int] = Field(..., title="A list of narration ids that mention the instructions.")
    explanations: str = Field(..., title="The justifications and reasons presented in the tutorial for performing the step.")
    explanations_content_ids: list[int] = Field(..., title="A list of narration ids that mention the explanations.")
    tips: str = Field(..., title="The tips presented in the tutorial that can help in completing the step easier, faster, or more efficiently.")
    tips_content_ids: list[int] = Field(..., title="A list of narration ids that mention the tips.")
    # warnings: str = Field(..., title="The warnings presented in the tutorial that can help avoid mistakes when performing the step.")
    # warnings_content_ids: list[int] = Field(..., title="A list of narration ids that mention the warnings.")


class StructuredSubgoalSummarySchema(BaseModel):
    # "materials", "outcome", "setting", "instructions", "explanation", "tips", "tools"

    ### context
    # subgoal: str = Field(..., title="The detailed description of the subgoal/stage based on the narration.")
    # subgoal_content_ids: list[int] = Field(..., title="A list of narration ids that mention the subgoal.")
    steps: list[StepSchema] = Field(..., title="A comprehensive list of steps to complete the subgoal/stage. It does not have to be explicitly mentioned in the narration.")

    materials: list[ObjectSchema] = Field(..., title="A comprehensive list of materials or ingredients used in the tutorial to complete the subgoal/stage.It does not have to be explicitly mentioned in the narration.")

    tools: list[ObjectSchema] = Field(..., title="A comprehensive list of tools or equipments used in the tutorial to complete the the subgoal/stage.")

    outcomes: list[ObjectSchema] = Field(..., title="A comprehensive list of the outcomes or results of completing the subgoal/stage.")
    # setting: str = Field(..., title="The context in which the step is performed, including the location, time, and other relevant details.")
    # setting_content_ids: list[int] = Field(..., title="A list of narration ids that mention the setting.")