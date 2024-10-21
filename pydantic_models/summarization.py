from pydantic import BaseModel, Field, validator

class MetaInformationSchema(BaseModel):
    goal: str = Field(..., title="The main purpose of the video, including a brief description.")
    procedure: str = Field(..., title="A concise summary of the overall process to achieve the goal, reflecting the order shown in the video. This should be high-level, as specific steps will be detailed later.")
    outcome: str = Field(..., title="A brief description of the final outcomes/results of the procedure.")

class SubgoalInformationSchema(BaseModel):
    subgoal: str = Field(..., title="A concise statement of the specific objective to be achieved in this step.")
    context: str = Field(..., title="A brief summary of relevant outcomes from previous steps necessary to accomplish this subgoal.")
    materials: str = Field(..., title="A list of materials, tools, or equipment required to complete the subgoal.")
    instructions: str = Field(..., title="A concise list of step-by-step actions mentioned in the video. Leave blank if not mentioned in the video.")
    rationale: str = Field(..., title="The reasons or explanations specified in the video behind the required actions explaining why they are important or necessary. Leave blank if not mentioned in the video.")
    tips: str = Field(..., title="Additional advice, warnings, or background information mentioned in the video that can help in achieving the subgoal more effectively. Leave blank if not mentioned in the video.")
    outcome: str = Field(..., title="A brief description of what is achieved or created upon completing the subgoal.")

class CoarseInformationSchema(BaseModel):
    goal: str = Field(..., title="A detailed summary of the specific objective to be achieved by the procedure in the video.")
    context: str = Field(..., title="A brief summary of the overall setting and prerequisities necessary to succesfully complete the procedure.")
    materials: str = Field(..., title="A list of materials, tools, or equipment required to complete the procedure in the video.")
    instructions: str = Field(..., title="A concise list of step-by-step actions mentioned in the video. Leave blank if not mentioned in the video.")
    rationale: str = Field(..., title="The reasons or explanations specified in the video behind the required actions explaining why they are important or necessary. Leave blank if not mentioned in the video.")
    tips: str = Field(..., title="Additional advice, warnings, or background information mentioned in the video that aid task understanding or execution. Leave blank if not mentioned in the video.")
    result: str = Field(..., title="A brief description of what is achieved or created upon completing the procedure.")

class MetaSummarySchema(BaseModel):
    goal: str = Field(..., title="The main purpose of the video, including a brief description.")
    goal_quotes: list[str] = Field(..., title="A list of quotes from the video that describe the goal.")
    procedure: str = Field(..., title="A concise summary of the overall process to achieve the goal, reflecting the order shown in the video. This should be high-level, as specific steps will be detailed later.")
    procedure_quotes: list[str] = Field(..., title="A list of quotes from the video that describe the procedure.")
    outcome: str = Field(..., title="A brief description of the final outcomes/results of the procedure.")
    outcome_quotes: list[str] = Field(..., title="A list of quotes from the video that describe the outcome.")


class SubgoalSummarySchema(BaseModel):
    subgoal: str = Field(..., title="The 3-word summary of the specific objective to be achieved in this subgoal.")
    subgoal_quotes: list[str] = Field(..., title="A list of quotes from the video that describe the subgoal.")
    materials: str = Field(..., title="A list of ingredients, materials, tools, or equipment required to complete the subgoal.")
    materials_quotes: list[str] = Field(..., title="A list of quotes from the video that describe the materials.")
    outcome: str = Field(..., title="A brief description of what is achieved or created upon completing the subgoal.")
    outcome_quotes: list[str] = Field(..., title="A list of quotes from the video that describe the outcome.")
    instructions: str = Field(..., title="A concise list of step-by-step actions mentioned in the video. Leave blank if not mentioned in the video.")
    instructions_quotes: list[str] = Field(..., title="A list of quotes from the video that describe the instructions.")
    rationale: str = Field(..., title="The reasons or explanations specified in the video behind the required actions explaining why they are important or necessary. Leave blank if not mentioned in the video.")
    rationale_quotes: list[str] = Field(..., title="A list of quotes from the video that describe the rationale.")
    tips: str = Field(..., title="Additional advice, warnings, or background information mentioned in the video that can help in achieving the subgoal more effectively. Leave blank if not mentioned in the video.")
    tips_quotes: list[str] = Field(..., title="A list of quotes from the video that describe the tips.")


class ProblemSummarySchema(BaseModel):
    goal: str = Field(..., title="The description of the specific objective to be achieved in the video.")
    goal_quotes: list[str] = Field(..., title="A list of quotes from the video that refer to or describe the goal.")
    
    objects: str = Field(..., title="A list of all necessary objects, ingredients, and materials required to achieve the goal.")
    objects_quotes: list[str] = Field(..., title="A list of quotes from the video that refer to or describe the objects.")
    
    outcome: str = Field(..., title="A detailed description of what is achieved or created upon completing the goal.")
    outcome_quotes: list[str] = Field(..., title="A list of quotes from the video that describe the outcome.")
    
class MethodSummarySchema(BaseModel): 
    instructions: str = Field(..., title="A step-by-step instructions mentioned in the video.")
    instructions_quotes: list[str] = Field(..., title="A list of quotes from the video that describe the instructions.")
    
    rationale: str = Field(..., title="The reasons or explanations for the steps specified in the video.")
    rationale_quotes: list[str] = Field(..., title="A list of quotes from the video that describe the rationale.")
    
    tips: str = Field(..., title="Additional advice, warnings, or background information mentioned in the video that can help in achieving the goal or following the instructions.")
    tips_quotes: list[str] = Field(..., title="A list of quotes from the video that describe the tips.")