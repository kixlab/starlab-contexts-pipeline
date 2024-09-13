from pydantic import BaseModel, Field, validator

from enum import Enum

class Relation(BaseModel):
    reference1: list[str] = Field(..., title="the list of relevant quotes from the first source.")
    reference2: list[str] = Field(..., title="the list of relevant quotes from the second source.")
    description: str = Field(..., title="the concise description of the relation between the two sources.")
    reasoning: str = Field(..., title="the reasoning behind the relation.")
    implication: str = Field(..., title="the potential implications of the relation.")
    helpfulness: str = Field(..., title="the potential helpfulness of the relation to the learners.")

    def get_dict(self):
        return {
            "reference1": self.reference1,
            "reference2": self.reference2,
            "description": self.description,
            "reasoning": self.reasoning,
            "implication": self.implication,
        }

class InformationRelations(BaseModel):
    consistent: list[Relation] = Field(..., title="The list of consistent information between sources.")
    complementary: list[Relation] = Field(..., title="The list of complementary information between sources.")
    contradictory: list[Relation] = Field(..., title="The list of contradictory information between sources.")

    def get_dict(self):
        return {
            "consistent": [r.get_dict() for r in self.consistent],
            "complementary": [r.get_dict() for r in self.complementary],
            "contradictory": [r.get_dict() for r in self.contradictory],
        }

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


class InformationAlignmentSchema(BaseModel):
    newref: list[str] = Field(..., title="the list of relevant quotes from the new video.")
    prevref: list[str] = Field(..., title="(if any) the list of relevant quotes from the previous video.")
    description: str = Field(..., title="the concise description of the new content in the new video.")

    def get_dict(self):
        return {
            "description": self.description,
            "newref": self.newref,
            "prevref": self.prevref,
        }

class AlignmentsSchema(BaseModel):
    alignments: list[InformationAlignmentSchema] = Field(..., title="The comprehensive and exhaustive list of new content in the new video")

    def get_dict(self):
        return {
            "alignments": [a.get_dict() for a in self.alignments],
        }

class InformationEnum(str, Enum):
    additional_information = "`Additional Information`"
    alternative_method = "`Alternative Method`"
    alternative_setting = "`Alternative Setting`"
    alternative_example = "`Alternative Example`"

    
class AlignmentClassificationSchema(BaseModel):
    classification: InformationEnum = Field(..., title="the classification of the new information into one of (1) additional_information, (2) alternative method, (3) alternative_setting, (4) alternative_example.")
    explanation: str = Field(..., title="the explanation of the classification.")
    provenance: str = Field(..., title="the provenance of the new information. why is the new content present only in the new video?")
    helpfulness: str = Field(..., title="the helpfulness of the new content to learners in improving their understanding.")

    def get_dict(self):
        return {
            "provenance": self.provenance,
            "helpfulness": self.helpfulness,
            "classification": self.classification,
            "explanation": self.explanation,
        }

class ReferenceAlignment(BaseModel):
    alignment_id: str = Field(..., title="the id of the relevant alignment.")
    description: str = Field(..., title="the concise description of the information alignment.")
    
    def get_dict(self):
        return {
            "alignment_id": self.alignment_id,
            "description": self.description,
        }

class AlignmentHookSchema(BaseModel):
    newref: list[str] = Field(..., title="the list of relevant quotes from the new video.")
    alignments: list[ReferenceAlignment] = Field(..., title="The list of alignments between the new video and the previous videos that pertain to `newref`.")
    title: str = Field(..., title="the hook that describes the alignments and can be used to interest learners in the new content or overall comparison.")

    def get_dict(self):
        return {
            "alignments": [a.get_dict() for a in self.alignments],
            "title": self.title,
            "newref": self.newref,
        }
    
class AlignmentHooksSchema(BaseModel):
    hooks: list[AlignmentHookSchema] = Field(..., title="The comprehensive and exhaustive list of hooks cover all information alignments")

    def get_dict(self):
        return {
            "hooks": [h.get_dict() for h in self.hooks],
        }