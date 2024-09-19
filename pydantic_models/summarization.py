from pydantic import BaseModel, Field, validator

class MetaInformationSchema(BaseModel):
    goal: str = Field(..., title="The main purpose of the video, including a brief description.")
    procedure: str = Field(..., title="A concise summary of the overall process to achieve the goal, reflecting the order shown in the video. This should be high-level, as specific steps will be detailed later.")
    outcome: str = Field(..., title="A brief description of the final outcomes/results of the procedure.")

    # def get_dict(self):
    #     return {
    #         "problem": {
    #             "goal": self.goal,
    #         },
    #         "method": {
    #             "procedure": self.procedure,
    #         },
    #         "result": {
    #             "result": self.result,
    #         },
    #     }

    def get_dict(self):
        return {
            "goal": self.goal,
            "procedure": self.procedure,
            "outcome": self.outcome,
        }

class SubgoalInformationSchema(BaseModel):
    subgoal: str = Field(..., title="A concise statement of the specific objective to be achieved in this step.")
    context: str = Field(..., title="A brief summary of relevant outcomes from previous steps necessary to accomplish this subgoal.")
    materials: str = Field(..., title="A list of materials, tools, or equipment required to complete the subgoal.")
    instructions: str = Field(..., title="A concise list of step-by-step actions mentioned in the video. Leave blank if not mentioned in the video.")
    rationale: str = Field(..., title="The reasons or explanations specified in the video behind the required actions explaining why they are important or necessary. Leave blank if not mentioned in the video.")
    tips: str = Field(..., title="Additional advice, warnings, or background information mentioned in the video that can help in achieving the subgoal more effectively. Leave blank if not mentioned in the video.")
    outcome: str = Field(..., title="A brief description of what is achieved or created upon completing the subgoal.")

    def get_dict(self):
        return {
            "subgoal": self.subgoal,
            "context": self.context,
            "materials": self.materials,
            "instructions": self.instructions,
            "rationale": self.rationale,
            "tips": self.tips,
            "outcome": self.outcome,
        }

    # def get_dict(self):
    #     return {
    #         "problem": {
    #             "subgoal": self.subgoal,
    #             "context": self.context,
    #         },
    #         "method": {
    #             "materials": self.materials,
    #             "instructions": self.instructions,
    #             "rationale": self.rationale,
    #             "tips": self.tips,    
    #         },
    #         "result": {
    #             "result": self.result,
    #         }
    #     }

class CoarseInformationSchema(BaseModel):
    goal: str = Field(..., title="A detailed summary of the specific objective to be achieved by the procedure in the video.")
    context: str = Field(..., title="A brief summary of the overall setting and prerequisities necessary to succesfully complete the procedure.")
    materials: str = Field(..., title="A list of materials, tools, or equipment required to complete the procedure in the video.")
    instructions: str = Field(..., title="A concise list of step-by-step actions mentioned in the video. Leave blank if not mentioned in the video.")
    rationale: str = Field(..., title="The reasons or explanations specified in the video behind the required actions explaining why they are important or necessary. Leave blank if not mentioned in the video.")
    tips: str = Field(..., title="Additional advice, warnings, or background information mentioned in the video that aid task understanding or execution. Leave blank if not mentioned in the video.")
    result: str = Field(..., title="A brief description of what is achieved or created upon completing the procedure.")

    def get_dict(self):
        return {
            "goal": self.goal,
            "context": self.context,
            "materials": self.materials,
            "instructions": self.instructions,
            "rationale": self.rationale,
            "tips": self.tips,
            "result": self.result,
        }


class MetaSummarySchema(BaseModel):
    goal: str = Field(..., title="The main purpose of the video, including a brief description.")
    procedure: str = Field(..., title="A concise summary of the overall process to achieve the goal, reflecting the order shown in the video. This should be high-level, as specific steps will be detailed later.")
    outcome: str = Field(..., title="A brief description of the final outcomes/results of the procedure.")

    # def get_dict(self):
    #     return {
    #         "problem": {
    #             "goal": self.goal,
    #         },
    #         "method": {
    #             "procedure": self.procedure,
    #         },
    #         "result": {
    #             "result": self.result,
    #         },
    #     }

    def get_dict(self):
        return {
            "goal": self.goal,
            "procedure": self.procedure,
            "outcome": self.outcome,
        }

class SubgoalSummarySchema(BaseModel):
    title: str = Field(..., title="The title of the subgoal.")
    materials: str = Field(..., title="A list of ingredients, materials, tools, or equipment required to complete the subgoal.")
    outcome: str = Field(..., title="A brief description of what is achieved or created upon completing the subgoal.")
    instructions: str = Field(..., title="A concise list of step-by-step actions mentioned in the video. Leave blank if not mentioned in the video.")
    rationale: str = Field(..., title="The reasons or explanations specified in the video behind the required actions explaining why they are important or necessary. Leave blank if not mentioned in the video.")
    tips: str = Field(..., title="Additional advice, warnings, or background information mentioned in the video that can help in achieving the subgoal more effectively. Leave blank if not mentioned in the video.")

    def get_dict(self):
        return {
            "subgoal": self.subgoal,
            "materials": self.materials,
            "instructions": self.instructions,
            "rationale": self.rationale,
            "tips": self.tips,
            "outcome": self.outcome,
        }
    
class SubgoalSummariesSchema(BaseModel):
    subgoal_summaries: list[SubgoalSummarySchema] = Field(..., title="The list of subgoal summaries for each subgoal in the video.")

    def get_dict(self):
        return [subgoal_summary.get_dict() for subgoal_summary in self.subgoal_summaries]