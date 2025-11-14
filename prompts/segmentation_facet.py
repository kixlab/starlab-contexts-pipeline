from pydantic_models.framework import CandidateSegmentationFacetsSchema
from prompts import separate_pieces_to_str, segmentation_candidates_gen_to_struct, candidates_to_str

SYSTEM_PROMPT_DESCRIBE_CONTEXTS = """You are a helpful assistant who can analyze and describe the contexts where knowledge applies or does not apply for task {task}."""

### TODO: try w/o types of aspects
POSSIBLE_TYPES_OF_ASPECTS = """### POSSIBLE TYPES OF ASPECTS:
| Type | Example Titles | Example Labels | Example Distinction |
|------|----------------|----------------|----------------------|
| When | Stage of process | Setup / Execution | Different steps in time |
| Why | Purpose / Subgoal | Collect Data / Analyze Data | Different goals or intentions |
| Where | Environment | Field / Lab | Different physical or digital settings |
| What | Object of focus | Hardware / Software | Working on different components |
| How | Method / Tool used | Manual / Automated | Using different approaches or tools |"""

USER_PROMPT_FORM_SEGMENTATION_FACET_CANDIDATES = """
Given information pieces from several tutorial videos, identify a set of task context aspects (i.e., different kinds of temporal segmentations) that would assign DIFFERENT segment labels to the pieces. Follow the requirements and the procedure below.

### REQUIREMENTS
- Each proposed aspect should be a distinct temporal segmentation of the tutorial-style transcript that segments the transcript into meaningful segments. Its segmentation guidelines should ensure that the segmentation is non-overlapping, covers the entire transcript, and each segment must be labeled with only one segment label (i.e., no multi-label classification).
- It should be possible to find a unique signature for each given piece of information (i.e., the combination of aspects uniquely discriminates each piece of information from others).
- The aspects should be orthogonal (i.e., do not overlap semantically).
- Keep aspect titles short, but interpretable without additional context.
- Keep example segment labels short, but interpretable without additional context.

### PROCEDURE
1. Identify at least one aspect of a task context that would assign DIFFERENT segment labels to the given pieces of information.
    - Classify each aspect into one of the possible types of aspects: "when", "why", "where", "what", "how".
    - Briefly justify how the aspect would assign different segment labels to the given pieces of information and the choice of the type of aspect.
    - Provide a detailed definition of the aspect.
    - Provide guidelines that explain how to segment a tutorial-style transcript according to this aspect.
    - Provide a few examples of "segment labels" (e.g., label for each given piece of information)
2. If there are multiple aspects, ensure that they are orthogonal (i.e., do not overlap semantically) and that the combination of aspects uniquely discriminates each piece of information from others (i.e., each piece of information receives a unique combination of segment labels across the aspects).

### INPUTS
- Information pieces with their respective context extracted from potentially different tutorial videos about the task:
{pieces}

### OUTPUT
Return a list of task context aspects that satisfy the requirements."""

def form_segmentation_facet_candidates_request(task, pieces, generation_model, **kwargs):

    pieces_str = separate_pieces_to_str(pieces)

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_DESCRIBE_CONTEXTS.format(task=task),
        },
        {
            "role": "user",
            "content": USER_PROMPT_FORM_SEGMENTATION_FACET_CANDIDATES.format(task=task, 
            pieces=pieces_str),
        },
    ]
    return {
        "messages": messages,
        "response_format": CandidateSegmentationFacetsSchema,
        "model": generation_model,
    }

def form_segmentation_facet_candidates_response(response, **kwargs):
    return segmentation_candidates_gen_to_struct(response["candidates"])



USER_PROMPT_COMBINE_SEGMENTATION_FACET_CANDIDATES = """
Given a list of aspects of a task context (i.e., each is different kind of temporal segmentation) combine them into a smaller, orthogonal, and comprehensive set of aspects. Follow the procedure below.

### PROCEDURE
Repeat until exit condition is met:
1. Check for segmentation similarity (i.e., the segmentation wrt one aspect is similar to the segmentation wrt another aspect):
    - Combine aspects that are similar into a single aspect.
2. Check for orthogonality (i.e., the segmentation wrt one aspect does not determine the segmentation wrt another aspect):
    - Redefine or replace non-orthogonal pairs with independent aspects of a task context.
3. Check for single-slot (i.e., the segmentation title and labels are short and easily interpretable):
    - Rephrase aspects of a task context so that segmentation title and labels fit in max. 3 words, but are easily interpretable.
4. Check for coverage (i.e., the new set preserves all the meaning of the original aspects of a task context):
    - Ensure the new set preserves all the meaning of the original aspects of a task context.
exit) The aspect set is smallest (i.e., aspects cannot be combined anymore), orthogonal, single-slot, and covers all the initial aspects.

### INPUTS
- Aspects of a task context (e.g., "[F1] ..."):
{candidates}

### OUTPUT
Return the final combined set of task context aspects."""

def combine_segmentation_facet_candidates_request(task, candidates, generation_model, **kwargs):
    candidates_str = candidates_to_str(candidates)

    messages = [

        {
            "role": "system",
            "content": SYSTEM_PROMPT_DESCRIBE_CONTEXTS.format(task=task),
        },
        {
            "role": "user",
            "content": USER_PROMPT_COMBINE_SEGMENTATION_FACET_CANDIDATES.format(candidates=candidates_str),
        },
    ]
    return {
        "messages": messages,
        "response_format": CandidateSegmentationFacetsSchema,
        "model": generation_model,
    }


def combine_segmentation_facet_candidates_response(response, **kwargs):
    return segmentation_candidates_gen_to_struct(response["candidates"])