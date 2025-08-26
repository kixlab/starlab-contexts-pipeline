from pydantic import BaseModel, Field


### Extract Steps
class StepSchema(BaseModel):
    step: str = Field(..., title="The step for the user to follow.")
    description: str = Field(..., title="The detailed description of the step.")


class StepListSchema(BaseModel):
    steps: list[StepSchema] = Field(..., title="The complete list of steps.")

### Extract Steps & Phases Taxonomy
class PhaseSchema(BaseModel):
    phase: str = Field(..., title="The phase of the task.")
    steps: list[StepSchema] = Field(..., title="The representative steps in the phase.")

class TaxonomySchema(BaseModel):
    phases: list[PhaseSchema] = Field(..., title="The list of phases in the task.")

### Extract IPOs
class ItemSchema(BaseModel):
    name: str = Field(..., title="The name of the item.")
    description: str = Field(..., title="The description of the item.")
    inferred: bool = Field(..., title="Whether the items is not directly mentioned in the instructional text.")
    alternatives: list[str] = Field(..., title="If mentioned, the alternative items that can be used instead of the item.")

class IPOSchema(BaseModel):
    step: str = Field(..., title="The step for the user to follow.")
    present: bool = Field(..., title="Whether the step is present in the instructional text.")
    instructions: list[str] = Field(..., title="The instructions for the step (e.g., instructions, directions, etc.)")
    inputs: list[str] = Field(..., title="The list of all distinct inputs for the step (e.g., ingredients, materials, tools, conditions, etc.)")
    outputs: list[str] = Field(..., title="The list of all distinct outputs for the step (e.g., product, result, effects, etc.)")

class IPOListSchema(BaseModel):
    ipos: list[IPOSchema] = Field(..., title="The list of IPOs per stepfor the task.")


### Taxonomize IPOs
class IPOCategorySchema(BaseModel):
    name: str = Field(..., title="The name of the category.")
    description: str = Field(..., title="The description of the category.")
    examples: list[str] = Field(..., title="Five representative examples of the category.")

class IPOTaxonomySchema(BaseModel):
    inputs: list[IPOCategorySchema] = Field(..., title="The taxonomy of inputs.")
    methods: list[IPOCategorySchema] = Field(..., title="The taxonomy of methods.")
    outputs: list[IPOCategorySchema] = Field(..., title="The taxonomy of outputs.")


class WithReferenceSchema(BaseModel):
    # text: str = Field(..., title="A piece of information from the tutorial (e.g., description, instruction, explanation, or tip). Must be a direct quote from the tutorial or its paraphrase.")
    ref: str = Field(..., title="The actual reference to the information from the tutorial. Must be a direct quote from the tutorial.")
    text: str = Field(..., title="A piece of information from the tutorial (e.g., explanation or tip). Must be a complete sentence, a direct quote from the tutorial or its paraphrase. Do not engage in interpreting the information.")

class ItemInformationSchema(BaseModel):
    name: str = Field(..., title="The name of the `subject of interest` from inputs or outputs listed in the taxonomy.")
    present: bool = Field(..., title="Whether the `subject of interest` is present in the tutorial.")
    description: str = Field(..., title="The detailed description of the `subject of interest` according to the tutorial. Must consist of a list of nouns possibly with its descriptors (e.g., amount, state, feel, etc).")


class MethodInformationSchema(BaseModel):
    name: str = Field(..., title="The name of the `subject of interest` from methods listed in the taxonomy.")
    present: bool = Field(..., title="Whether the `subject of interest` is present in the tutorial.")
    instruction: str = Field(..., title="The detailed instruction about the `subject of interest` according to the tutorial. Must be a complete sentence, a direct quote from the tutorial or its paraphrase.")
    explanations: list[WithReferenceSchema] = Field(..., title="The explanation about the `subject of interest` (e.g., reasons why it was performed and its consequences) according to the tutorial.")
    tips: list[WithReferenceSchema] = Field(..., title="The tips (e.g., enhance the efficiency, improve the quality, etc.) or warnings (e.g., actions to avoid, etc.) about the `subject of interest` according to the tutorial.")
    
    ### no reference
    # instruction: str = Field(..., title="The detailed instruction about the `subject of interest` according to the tutorial.")
    # explanation: list[str] = Field(..., title="The explanation about the `subject of interest` (e.g., reasons why it was performed and its consequences) according to the tutorial. Can be a direct quote from the tutorial or its paraphrase. Must be a complete sentence.")
    # tips: list[str] = Field(..., title="The tips (e.g., enhance the efficiency, improve the quality, etc.) or warnings (e.g., actions to avoid, etc.) about the `subject of interest` according to the tutorial. Can be a direct quote from the tutorial or its paraphrase. Must be a complete sentence.")

# ### Extract Information per IPO
# class SubjectInformationSchema(BaseModel):
#     name: str = Field(..., title="The name of the subject of interest.")
#     present: bool = Field(..., title="Whether the subject is present in the tutorial.")
#     description: str = Field(..., title="The detailed description of the subject according to the tutorial.")
#     # description_reference: str = Field(..., title="The reference to the description of the subject in the tutorial.")
#     explanation: str = Field(..., title="The explanation about the subject according to the tutorial (e.g., reasons why it was performed and its consequences).")
#     # explanation_reference: str = Field(..., title="The reference to the explanation of the subject in the tutorial.")
#     tips: str = Field(..., title="The tips/warnings about the subject according to the tutorial.") 
#     # tips_reference: str = Field(..., title="The reference to the tips/warnings of the subject in the tutorial.")
#     # alternatives: str = Field(..., title="Any useful information about the subject according to the tutorial (e.g., alternatives, substitutes, etc.).) 
#     # alternatives_reference: str = Field(..., title="The reference to the alternatives of the subject in the tutorial.")

class IPOInformationSchema(BaseModel):
    inputs: list[ItemInformationSchema] = Field(..., title="The list of detailed information about inputs.")
    methods: list[MethodInformationSchema] = Field(..., title="The list of detailed information about methods.")
    outputs: list[ItemInformationSchema] = Field(..., title="The list of detailed information about outputs.")

class ClusterInformationSchema(BaseModel):
    representative: str = Field(..., title="The representative information piece of the cluster.")
    ids: list[int] = Field(..., title="The ids of the information pieces that are in the cluster.")

class ClusterListSchema(BaseModel):
    clusters: list[ClusterInformationSchema] = Field(..., title="The list of clusters.")