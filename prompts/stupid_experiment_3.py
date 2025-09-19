import json
from helpers import get_response_pydantic

from pydantic_models.experiment_3 import SegmentationSchema

from pydantic_models.experiment_3 import InformationPiecesSchema

from pydantic_models.experiment_3 import AnswerListSchema

from pydantic_models.experiment_3 import LabeledPiecesSchema

from pydantic_models.experiment_3 import CandidateFacetsSchema

TAXONOMY = {
    "opening": "Starting remarks and instructor/channel introductions",
    "closing": "Parting remarks and wrap-up",
    "goal": "Main purpose of the video and its descriptions",
    "motivation": "Reasons or background information on why the video was created",
    "briefing": "Rundown of how the goal will be achieved",
    "subgoal": "Objective of a subsection",
    "instruction": "Actions that the instructor performs to complete the task",
    "tool": "Introduction of the materials, ingredients, and equipment to be used",
    "tip": "Additional instructions or information that makes instructions easier, faster, or more efficient",
    "warning": "Actions that should be avoided",
    "justification": "Reasons why the instruction was performed",
    "effect": "Consequences of the instruction",
    "status": "Descriptions of the current state of the target object",
    "context": "Descriptions of the method or the setting",
    "tool specification": "Descriptions of the tools and equipment",
    "outcome": "Descriptions of the final results of the procedure",
    "reflection": "Summary, evaluation, and suggestions for the future about the overall procedure",
    "side note": "Personal stories, jokes, user engagement, and advertisements",
    "self-promotion": "Promotion of the instructor of the channel (i.e. likes, subscription, notification, or donations)",
    "bridge": "Meaningless phrases or expressions that connect different sections",
    "filler": "Conventional filler words",
    "other": "None of the specfied categories",
}


SYSTEM_PROMPT_PROCESS_TRANSCRIPT = """
You are a helpful assistant who can understand and analyze tutorial videos."""

USER_PROMPT_FORM_INFORMATION_UNITS = """
You are analyzing a tutorial video for {task}.

From a tutorial-style transcript (recipe, SOP, repair guide, etc.) produce a structured, machine-readable representation of every atomic information piece. Work strictly in the order below.

1. Detect atomic information pieces
- Parse the transcript clause-by-clause.
- Create one piece for each indivisible action, object, reason, or tip.
    - Indivisible = removing any word breaks the meaning.
- If a sentence contains multiple actions (`whisk, then fold`) or multiple rationales, split them.
- Rewrite each piece so it is understandable standing alone—no dangling `it`,`they`, `this step`, etc.
Examples:
- `Whisk the eggs for 30 s, then fold in the flour.` -> `Whisk the eggs for 30 seconds.` `Fold in the flour.`
- `Let the paint dry so it won't smudge.` -> `Let the paint dry.` (method) `This prevents smudging.` (explanation)
- `Add the beaten eggs to the mixture and mix well.` -> `Add the beaten eggs to the mixture.` `Mix well.`
- `Add 1 cup of white chocolate chips and stir until thoroughly combined.` -> `Add 1 cup of white chocolate chips.` `Stir the mixture until thoroughly combined.`

2. Assign `content_type` to each piece: `Greeting`, `Overview`, `Method`, `Supplementary`, `Explanation`, `Description`, `Conclusion`, and `Miscellaneous`. Do not add sub category!
- Greeting
Opening: Starting remarks and instructor/channel introductions.
Example: "Hey, what's up you guys, Chef [...] here."
Closing: Parting remarks and wrap-up.
Example: "Stay tuned, we'll catch you all later."
- Overview
Goal: Main purpose of the video and its descriptions.
Example: "Today, I'll show you a special technique which is totally special and about image pressing."
Motivation: Reasons or background information on why the video was created.
Example: "[...] Someone is making a very special valentine's day meal for another certain special someone."
Briefing: Rundown of how the goal will be achieved.
Example: "I'm pretty sure that just taking a pencil and putting it over the front and then putting a bunch of rubber bands around the pencil
[...] that's going to do it."
- Method
Subgoal: Objective of a subsection.
Example: "Now for the intricate layer that will give me the final webbing look."
Instruction: Actions that the instructor performs to complete the task.
Example: "We're going to pour that into our silicone baking cups."
Tool: Introduction of the materials, ingredients, and equipment to be used.
Example: "I'm also going to use a pair of scissors, a glue stick, some fancy tape or some regular tape."
- Supplementary
Tip: Additional instructions or information that makes instructions easier, faster, or more efficient.
Example: "I find that it's easier to do just a couple of layers at a time instead of all four layers at a time."
Warning: Actions that should be avoided.
Example: "I don't know but I would say avoid using bleach if you can."
- Explanation
Justification: Reasons why the instruction was performed.
Example: "Because every time we wear our contact lenses, makeup and even dirt particles [...] might harm our eyes directly."
Effect: Consequences of the instruction.
Example: "And these will overhang a little to help hide the gap."
- Description
Status: Descriptions of the current state of the target object.
Example: "Something sticky and dirty all through the back seat."
Context: Descriptions of the method or the setting.
Example: "[...] The process of putting on a tip by hand [...] takes a lot of patience but it can be done if you're in a pinch."
Tool Specification: Descriptions of the tools and equipment.
Example: "These are awesome beans, creamy texture, slightly nutty loaded with flavor."
- Conclusion
Outcome: Descriptions of the final results of the procedure.
Example: "And now we have a dinosaur taggy blanket that wrinkles, so a fun gift for any baby on your gift giving list."
Reflection: Summary, evaluation, and suggestions for the future about the overall procedure.
Example: "However, I am still concerned about how safe rubbing alcohol actually is to use so maybe next time, I will give vodka a try."
- Miscellaneous
Side Note: Personal stories, jokes, user engagement, and advertisements.
Example: "Tristan is back from basketball - He made it on the team so it's pretty exciting."
Self-promotion: Promotion of the instructor of the channel (i.e. likes, subscription, notification, or donations).
Example: "So if you like this video, please give it a thumbs up and remember to subscribe."
Bridge: Meaningless phrases or expressions that connect different sections.
Example: "And we're going to go ahead and get started."
Filler: Conventional filler words.
Example: "Whoops."

3. Time-stamps
- For video/audio transcripts, copy the exact start_time and end_time (in seconds) that bracket the source text of the item.
- If no timing metadata exists, output null for both.

The transcript with time-stamps (in seconds) is as follows: ```
{transcript}
```"""

def transcript_to_str(transcript, with_timestamps=True):
    if len(transcript) == 0:
        return "No transcript provided."
    
    transcript_strs = []
    for subtitle in transcript:
        if with_timestamps:
            start_str = f"{int(subtitle['start'])}"
            end_str = f"{int(subtitle['end'])}"
            transcript_strs.append(f"[{start_str} - {end_str}] {subtitle['text']}")
        else:
            transcript_strs.append(subtitle['text'])
    return "\n".join(transcript_strs)

def form_information_units(task, transcript):

    transcript_str = transcript_to_str(transcript)

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_PROCESS_TRANSCRIPT,
        },
        {
            "role": "user",
            "content": USER_PROMPT_FORM_INFORMATION_UNITS.format(task=task, transcript=transcript_str),
        },
    ]

    response = get_response_pydantic(messages, InformationPiecesSchema)

    pieces = response["pieces"]
    return pieces

USER_PROMPT_FORM_CONTEXT_CODEBOOK = """
You are updating a codebook for the {facet} facet of {task}. 
A "{facet}" is defined as a canonical answer to the question: "{question}".

### INPUTS
- Existing canonical {facet_plural}:
{answers}
- Tutorial-style transcript with timestamps (in seconds):
{transcript}    

### GOAL
Return an updated list of canonical {facet_plural} (including existing and any new ones needed to cover the given transcript), each with 1-2 concrete examples grounded in the given transcript (or reused from the existing list if not found in the transcript).

### GUIDELINES FOR CANONICAL ANSWERS:
{answer_guidelines}

### GENERAL NORMALIZATION RULES (apply in addition to the above):
- Merge synonyms/variants into one canonical value; keep the existing value as the canonical form when applicable.
- Use concise, unambiguous phrasing; avoid brand-specific or overly narrow wording unless essential to task fidelity.
- Only add a new {facet} if the transcript clearly contains an answer to "{question}" that is not covered by any existing canonical value.

### PROCEDURE
1) Scan the transcript for spans that directly answer "{question}".
2) For each span:
    - Map it to an existing canonical {facet} if it's the same concept (consider synonyms, pluralization, formatting).
    - If no existing value fits, propose a new canonical {facet} that follows the guidelines.
3) For every canonical {facet} in the final list (existing or new):
    - Provide 1-2 representative examples.
    - Each example must include:
        - content (the exact minimal excerpt that supports the answer)
        - context (around 10-20 words surrounding the content and the content itself)
    - If a canonical {facet} is not evidenced in the given transcript, reuse one example from the existing list if available.

### OUTPUT
Return a list of canonical answers. Each answer should have the following fields:
- id: id of the canonical {facet}
- answer: short representative value less than 2-3 words
- definition: elaboration of what the answer means
- examples: 1-2 short representative content and context that would be labeled as the answer

NOTES
- Ground everything in the provided inputs; no external knowledge or speculation.
- Do not create duplicate or overlapping canonical values.
- Keep examples short and specific; avoid paraphrasing for the 'content' field."""

ANSWER_FORMAT = """[{answer_id}] {answer_title}
Definition: {answer_definition}
Examples (Contents and Context): ```
{examples}
```"""

ANSWER_EXAMPLE_FORMAT = """\t- Content {example_idx}: {example_content}
\t- Context {example_idx}: {example_context}"""

def examples_to_str(examples):
    if len(examples) == 0:
        return "No examples contents and context are available."
    
    example_strs = []
    for example_idx, example in enumerate(examples):
        example_strs.append(ANSWER_EXAMPLE_FORMAT.format(example_context=example["context"], example_content=example["content"], example_idx=example_idx + 1))
    example_str = "\n".join(example_strs)
    return example_str

def answers_to_str(answers):
    if len(answers) == 0:
        return "No facet values (i.e., canonical answers) are available."

    answer_strs = []
    for answer_idx, answer in enumerate(answers):
        answer_id = f"A{answer_idx + 1}"
        answer_strs.append(ANSWER_FORMAT.format(answer_id=answer_id, answer_title=answer["answer"], answer_definition=answer["definition"], examples=examples_to_str(answer["examples"])))

    answer_str = "\n".join(answer_strs)
    return answer_str

def guidelines_to_str(guidelines):
    if len(guidelines) == 0:
        return "No guidelines are available."
    
    guideline_strs = []
    for guideline in guidelines:
        guideline_strs.append(f"- {guideline}")
    guideline_str = "\n".join(guideline_strs)
    return guideline_str


def form_codebook(task, transcript, facet):
    transcript_str = transcript_to_str(transcript)

    answer_guidelines_str = guidelines_to_str(facet["answer_guidelines"])

    answers_str = answers_to_str(facet["answers"])

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_PROCESS_TRANSCRIPT,
        },
        {
            "role": "user",
            "content": USER_PROMPT_FORM_CONTEXT_CODEBOOK.format(task=task, facet_plural=facet["title_plural"], facet=facet["title"], question=facet["question"], answer_guidelines=answer_guidelines_str, answers=answers_str, transcript=transcript_str)
        },
    ]

    response = get_response_pydantic(messages, AnswerListSchema)

    answers = response["answers"]
    for answer in answers:
        del answer["id"]
    return answers


USER_PROMPT_LABEL_TRANSCRIPT_PIECES = """
You are labeling a tutorial video for {task}.

### DEFINITION
- A "{facet}" is defined as a canonical answer to the question: "{question}".
- "{facet_plural}" refers to the full set of canonical answers provided below. You must choose only from this set.

### INPUTS
- Canonical {facet_plural} (codebook) (each starts with an ID in square brackets, e.g., "[A1] ..."):
{answers}
- Pieces of information (each starts with an ID in square brackets, e.g., "[p12] ..."):
{pieces}

### GOAL
For each piece, assign exactly one {facet} from the codebook when the content or context clearly answers the question "{question}". 
If no {facet} is clearly supported, assign the empty string "".

### MATCHING RULES
- Choose only from the provided canonical {facet_plural}; output the canonical string exactly as given.
- Treat synonyms/variants in the piece as mapping to the relevant canonical value; do not invent new values.
- Ignore negations, hypotheticals, and uncertain mentions (e.g., "maybe", "could", "if needed")—return "" in those cases.
- If multiple {facet_plural} seem plausible, pick the single best answer that is most specific to the piece's content.
- Do not use external knowledge. Ground decisions strictly in the provided transcript.

### OUTPUT
Return a list of labeled pieces. Each piece should have the following fields:
- piece_id: id of the piece
- answer_id: id of the canonical {facet} from the codebook or ""
- answer: canonical {facet} string or ""

### NOTES
- Preserve the original order of pieces.
- Do not drop or merge pieces.
"""

def pieces_to_str(pieces):
    pieces_strs = []
    for piece_idx, piece in enumerate(pieces):
        pieces_strs.append(f"[{piece_idx+1}] {piece['content']}")
    pieces_str = "\n".join(pieces_strs)
    return pieces_str

def label_transcript_pieces(task, pieces, facet):
    pieces_str = pieces_to_str(pieces)

    answers_str = answers_to_str(facet["answers"])
    
    if len(facet["answers"]) == 0:
        print("STRONG WARNING: No canonical answers found in the facet.")
        return []

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_PROCESS_TRANSCRIPT,
        },
        {
            "role": "user",
            "content": USER_PROMPT_LABEL_TRANSCRIPT_PIECES.format(task=task, pieces=pieces_str, facet_plural=facet["title_plural"], facet=facet["title"], question=facet["question"], answers=answers_str),
        },
    ]
    response = get_response_pydantic(messages, LabeledPiecesSchema)

    labeled_pieces = response["labeled_pieces"]
    piece_to_label = {}
    for labeled_piece in labeled_pieces:
        cur_piece_id = pieces[int(labeled_piece["piece_id"])-1]["piece_id"]
        if cur_piece_id in piece_to_label:
            print("STRONG WARNING: Multiple labels found for the same piece ID.", cur_piece_id)
        piece_to_label[cur_piece_id] = labeled_piece["answer"]

    formatted_pieces = []
    for piece in pieces:
        cur_piece_id = piece["piece_id"]
        if cur_piece_id in piece_to_label:
            formatted_pieces.append(piece_to_label[cur_piece_id])
        else:
            print("STRONG WARNING: No label found for the piece ID.", cur_piece_id)
            formatted_pieces.append("")

    return formatted_pieces


SYSTEM_PROMPT_FORM_FACET_CANDIDATES = """You are a helpful assistant who identifies/refines `applicability facets` that define why/when/where pieces of information about a procedural task apply."""

USER_PROMPT_FORM_FACET_CANDIDATES = """
You are analyzing tutorial-style information (recipe, SOP, repair guide, etc.) for {task}.

### INPUTS
- Information items:
{pieces}

### GOAL
You are given a list of information items. Provide a smallest list of applicability facets that can be used to differentiate all the provided information items (i.e., each semantically distinct item can be given a unique applicability signature across the facets). The facets should be:
- based on a single, concrete question about applicability of the information (why/when/where the information applies).
- orthogonal (the answer to one facet does not deterministically fix the answer to another).
- single-slot (the answer is one short phrase less than 3 words).

### PROCEDURE (iterate until stable)
1) Go over all the pairs of information items and check:
    a) if they are semantically similar, skip.
    b) if they are semantically distinct, try the following methods and pick the one that satisfy the main goal:
        - REUSE/EXTEND: See if the pair can be distinguished by at least one of the existing facets or by extending one of the existing facets; if necessary, refine the question to cover the distinction of the new pair.
        - ADD: Introduce exactly one new applicability facet (short facet title, question with less than 20 words, guidelines for the answer). The question should be about applicability of the information (why/when/where the information applies). The question should be single-slot (answerable with a short phrase less than 3 words) and orthogonal to others.
2) Check for orthogonality, single-slot, and "about applicability":
    - Check for orthogonality:
        - Redefine or replace non-orthogonal pairs with independent facets.  
        - Example (WRONG: not independent):  
            - [F3] In what place would this information be useful?  
            - [F4] What tools does this information apply to?
            - Tool applicability is deterministically tied to physical setting, so can focus on one of the two.
        - Example (CORRECT: independent redesign):  
            - [F4] What tools does this information apply to?
    - Check for single-slot:
        - Rephrase questions so answers fit in less than 3 words.
        - Example (WRONG: open-ended):  
            - [F5] Why does this information apply in this situation?  
        - Example (CORRECT: single-slot):  
            - [F5] What result/effect is this information trying to help achieve?
    - Check for "about applicability":
        - Ensure the question is about applicability of the information, not its description.
        - Example (WRONG: descriptive):  
                - [F6] Why pressing the button?
        - Example (CORRECT: applicability):  
                - [F6] What result/effect is this information trying to help achieve when pressing the button?

### OUTPUT
The list of APPLICABILITY FACETS, where each facet has the following fields:
- ID
- Type: Why | When | Where
- Title (less than 2-3 words)
- Plural title
- Question (less than 20 words)
- Guidelines on how to extract the answer from the information item/context (e.g., what to look for, what to ignore, format of the answer: length, how to canonize the answer)
- Example answers (1-2): short representative values 1-2 words

### NOTES
- Use domain-canonical labels/units. Try to ensure consistency in the format of the answers.
- Example answers are illustrative only, do not attempt completeness or invent unsupported values.
"""

CANDIDATE_FACET_FORMAT = """[{facet_id}] ({facet_type}) {facet_title} (Plural: {facet_title_plural})
Question formulation: {question} 
Answer guidelines: ```
{answer_guidelines}
```
Example answers: ```
{answers}
```"""


def candidates_gen_to_struct(gen_candidates):
    struct_candidates = []
    for candidate in gen_candidates:
        answers = []
        for answer in candidate["examples"]:
            answers.append({
                "answer": answer["answer"],
                "definition": answer["definition"],
                "examples": [],
            })
        struct_candidates.append({
            "type": candidate["type"],
            "title": candidate["title"],
            "title_plural": candidate["title_plural"],
            "question": candidate["question"],
            "answer_guidelines": candidate["answer_guidelines"],
            "answers": answers,
        })
    return struct_candidates

def candidates_to_str(candidates):
    candidates_strs = []
    for idx, candidate in enumerate(candidates):
        guideliens_str = guidelines_to_str(candidate["answer_guidelines"])
        answers_str = answers_to_str(candidate["answers"])
        candidates_strs.append(CANDIDATE_FACET_FORMAT.format(
            facet_id=f"F{idx+1}",
            facet_type=candidate["type"],
            facet_title=candidate["title"],
            facet_title_plural=candidate["title_plural"],
            question=candidate["question"],
            answer_guidelines=guideliens_str,
            answers=answers_str,
        ))
    candidates_str = "\n".join(candidates_strs)
    return candidates_str

def form_facet_candidates(task, pieces):
    pieces_str = pieces_to_str(pieces)

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_FORM_FACET_CANDIDATES,
        },
        {
            "role": "user",
            "content": USER_PROMPT_FORM_FACET_CANDIDATES.format(task=task, pieces=pieces_str),
        },
    ]
    response = get_response_pydantic(messages, CandidateFacetsSchema)

    # merged_candidates = combine_facet_candidates(
    #     task,
    #     current_candidates,
    #     candidates_gen_to_struct(response["candidates"]),
    # )

    # return merged_candidates
    return candidates_gen_to_struct(response["candidates"])


SYSTEM_PROMPT_COMBINE_FACET_CANDIDATES = """
You are a helpful assistant who can understand and analyze procedural knowledge and its applicability.
"""

USER_PROMPT_COMBINE_FACET_CANDIDATES = """
You are analyzing applicability facets (a single concrete question) that describe where knowledge about {task} applies. There are three general types of facets: Why (e.g., why this information applies), When (e.g., when this information applies), Where (e.g., where this information applies).

### INPUTS
- Applicability facets, each labeled with an ID in square brackets (e.g., "[F1] ..."):  
{candidates}

### GOAL
Combine the given facets to get a set of facets that:
- Cover all initial facets.
- Are orthogonal (the answer to one facet does not deterministically fix the answer to another).
- Are about applicability (the question concerns when/where/why a piece of information about the task applies).
- Are single-slot (the answer is one short phrase <3 words).

### PROCEDURE
Iterate until stable:
1) Check for duplicates:
    - Merge semantically equivalent facets.  
    - Example (WRONG: treated separately):  
        - [F1] What phase does this information apply?  
        - [F2] What time in the process does this information apply?  
    - Example (CORRECT: merged):  
        - [F1] What phase does this information apply?  
2) Check for orthogonality:
    - Redefine or replace non-orthogonal pairs with independent facets.  
    - Example (WRONG: not independent):  
        - [F3] In what place would this information be useful?  
        - [F4] What tools does this information apply to?
        - Tool applicability is deterministically tied to physical setting, so can focus on one of the two.
    - Example (CORRECT: independent redesign):  
        - [F4] What tools does this information apply to?
3) Check for single-slot:
    - Rephrase questions so answers fit in less than 3 words.
    - Example (WRONG: open-ended):  
        - [F5] Why does this information apply in this situation?  
    - Example (CORRECT: single-slot):  
        - [F5] What result/effect is this information trying to help achieve?
4) Check for "about applicability":
   - Ensure the question is about applicability of the information, not its description.
   - Example (WRONG: descriptive):  
        - [F6] Why pressing the button?
   - Example (CORRECT: applicability):  
        - [F6] What result/effect is this information trying to help achieve when pressing the button?
5) Check for coverage:
    - Ensure the new set preserves all the meaning of the original facets.

### EXIT CONDITION
The final facet set is smallest, orthogonal, single-slot, and fully covers the initial set. The facets should be about applicability of the information.

### OUTPUT
The list of APPLICABILITY FACETS, where each facet has the following fields:
- ID
- Type: Why | When | Where
- Title (less than 2-3 words)
- Plural title
- Question (less than 20 words)
- Guidelines on how to extract the answer from the information item/context (e.g., what to look for, what to ignore, format of the answer: length, how to canonize the answer)
- Example answers (1-2): short representative values 1-2 words

### NOTES
- Use domain-canonical labels/units.
- Example answers are illustrative, not exhaustive.
"""

def combine_facet_candidates(task, all_candidates):
    all_candidates_str = candidates_to_str(all_candidates)

    messages = [

        {
            "role": "system",
            "content": SYSTEM_PROMPT_COMBINE_FACET_CANDIDATES,
        },
        {
            "role": "user",
            "content": USER_PROMPT_COMBINE_FACET_CANDIDATES.format(task=task, candidates=all_candidates_str),
        },
    ]
    response = get_response_pydantic(messages, CandidateFacetsSchema)

    return candidates_gen_to_struct(response["candidates"])