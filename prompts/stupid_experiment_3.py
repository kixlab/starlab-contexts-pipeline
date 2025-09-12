import json
from helpers import get_response_pydantic

from pydantic_models.experiment_3 import SegmentationSchema

from pydantic_models.experiment_3 import InformationPiecesSchema

from pydantic_models.experiment_3 import LabelListSchema

from pydantic_models.experiment_3 import LabeledPiecesSchema

from pydantic_models.experiment_3 import CandidateConditionsSchema

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

The transcript with time-stamps (in seconds) is as follows:
```
{transcript}```"""

def form_information_units(task, transcript):

    transcript_str = ""
    for i, subtitle in enumerate(transcript):
        start_str = f"{int(subtitle['start'])}"
        end_str = f"{int(subtitle['end'])}"
        transcript_str += f"[{start_str} - {end_str}] {subtitle['text']}\n"

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
You are analyzing a tutorial video for {task}.
From a tutorial-style transcript (recipe, SOP, repair guide, etc.) extract the {schema_plural} involved in the task. A {schema} is {definition}.

Follow these guidelines when extracting {schema_plural}:
{guidelines}

First, review the existing list of {schema_plural} to identify if any of them are mentioned in the transcript. If so, use the same {schema_plural} names to ensure consistency whenever possible.
If you identify new {schema_plural} that are not in the existing list, add them appropriately at the end of the list along with the examples. An example should contain the content () and the context (some text (around 10-20 words) surrounding the content and the content itself).

Here is the existing list of {schema_plural}:
```
{labels}```

Here is the transcript with time-stamps (in seconds):
```
{transcript}```

Return a series of {schema_plural} as a list (both old and new). When providing the examples for each {schema}, try to ensure that there is a variety of examples (around 1-3) based on the given transcript or from the existing list of {schema_plural}."""

LABEL_FORMAT = """[{label_id}] {label_title}
Definition: {label_definition}
Examples: 
{examples}"""

LABEL_EXAMPLE_FORMAT = """\t- Content {example_idx}: {example_content}
\t- Context {example_idx}: {example_context}
"""

def labels_to_str(labels):
    labels_str = ""
    for label_idx, label in enumerate(labels):
        examples_str = ""
        for example_idx, example in enumerate(label["examples"]):
            examples_str += LABEL_EXAMPLE_FORMAT.format(example_context=example["context"], example_content=example["content"], example_idx=example_idx + 1)
        label_id = f"L{label_idx + 1}"
        labels_str += LABEL_FORMAT.format(label_id=label_id, label_title=label["title"], label_definition=label["definition"], examples=examples_str)
    return labels_str

def guidelines_to_str(guidelines):
    guidelines_str = ""
    for guideline in guidelines:
        guidelines_str += f"- {guideline}\n"
    return guidelines_str


def form_context_codebook(task, transcript, schema):
    transcript_str = ""
    for i, subtitle in enumerate(transcript):
        start_str = f"{int(subtitle['start'])}"
        end_str = f"{int(subtitle['end'])}"
        transcript_str += f"[{start_str} - {end_str}] {subtitle['text']}\n"

    guidelines_str = guidelines_to_str(schema["codebook_guidelines"])

    labels_str = labels_to_str(schema["labels"])
    
    if len(schema["labels"]) == 0:
        labels_str = f"No {schema['schema_plural']} yet. Define new ones.\n"

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_PROCESS_TRANSCRIPT,
        },
        {
            "role": "user",
            "content": USER_PROMPT_FORM_CONTEXT_CODEBOOK.format(task=task, schema_plural=schema["schema_plural"], schema=schema["schema"], definition=schema["definition"], guidelines=guidelines_str, labels=labels_str, transcript=transcript_str)
        },
    ]

    response = get_response_pydantic(messages, LabelListSchema)

    labels = response["labels"]
    for label in labels:
        del label["id"]
    return labels


USER_PROMPT_LABEL_TRANSCRIPT_PIECES = """
You are analyzing a tutorial video for {task}.
You are given a list of pieces of information from a tutorial (recipe, SOP, repair guide, etc.) along with a list of possible {schema_plural} involved in the task. A {schema} is {definition}.

Your task is to read through the pieces of information sequentially and label the pieces of information with the appropriate {schema}.
The {schema_plural} may not be in order in the list, and some {schema_plural} may not be used at all. Only assign a {schema} when the content clearly matches the {schema}. If it does not match any {schema}, leave it as empty string `""`.

Here is the list of {schema_plural}:
```
{labels}```

Here is the list of pieces of information with ids in square brackets `[]` (e.g., `[piece_id] content`):
```
{pieces}```

Return the pieces of information with labels in the same order as they were provided."""

def label_transcript_pieces(task, pieces, schema):
    pieces_str = ""
    for piece_idx, piece in enumerate(pieces):
        pieces_str += f"[{piece_idx+1}] {piece['content']}\n"

    labels_str = labels_to_str(schema["labels"])
    
    if len(schema["labels"]) == 0:
        print("STRONG WARNING: No labels found in the schema.")
        return []

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_PROCESS_TRANSCRIPT,
        },
        {
            "role": "user",
            "content": USER_PROMPT_LABEL_TRANSCRIPT_PIECES.format(task=task, pieces=pieces_str, schema_plural=schema["schema_plural"], schema=schema["schema"], definition=schema["definition"], labels=labels_str),
        },
    ]
    response = get_response_pydantic(messages, LabeledPiecesSchema)

    labeled_pieces = response["labeled_pieces"]
    piece_to_label = {}
    for labeled_piece in labeled_pieces:
        cur_piece_id = pieces[int(labeled_piece["piece_id"])-1]["piece_id"]
        if cur_piece_id in piece_to_label:
            print("STRONG WARNING: Multiple labels found for the same piece ID.", cur_piece_id)
        piece_to_label[cur_piece_id] = labeled_piece["label"]

    formatted_pieces = []
    for piece in pieces:
        cur_piece_id = piece["piece_id"]
        if cur_piece_id in piece_to_label:
            formatted_pieces.append(piece_to_label[cur_piece_id])
        else:
            print("STRONG WARNING: No label found for the piece ID.", cur_piece_id)
            formatted_pieces.append("")

    return formatted_pieces


SYSTEM_PROMPT_FORM_FACET_CANDIDATES = """
You are a helpful assistant who can identify important applicability conditions that can differentiate between different pieces of knowledge."""

USER_PROMPT_FORM_FACET_CANDIDATES = """
You are analyzing pieces of information from different tutorials for {task}.

You are given (a) pieces of information and (b) a list of candidate conditions that can differentiate where each of the pieces is applicable.

Condition Definition: A condition is a set of instances/values that can be used to describe the scope of applicability of a piece of knowledge. For example, if there is a piece of knowledge X (e.g., instruction) that applies at a step Y, then the condition is "step". and Y is the instance of the condition "step".

Create the smallest set of orthogonal conditions that together differentiate all conflicting pieces of information. It's fine if a condition ends up with 3-5 instances/values when that improves discrimination. Reuse and extend existing conditions first, then devise new ones only if needed. To ensure orthogonality, MERGE or REMOVE redundant conditions.

Process: REUSE, UPDATE, ADD, and MERGE/REMOVE;
1. Find conflicts: Identify pairs/groups of pieces that cannot apply together in the same context and note what distinguishes them.
2. Start from existing candidates:
    - REUSE exact matches.
    - If close, UPDATE the definition and/or guidelines so the condition now covers both the old and new meanings.
3. Only if necessary, ADD a new condition:
    - Decide if the condition should be method-based (i.e., "how/") or purpose-based (i.e., "why/outcome").
    - Ensure that the condition is concrete and concise.
4. MERGE/REMOVE redundant conditions:
    - If two conditions encode the same idea, MERGE them.
    - If a condition is irrelevant to the current set of pieces of information, REMOVE it.
In general, favor fewer conditions over fewer total instances/values.

Return the new list of relevant conditions.
For each condition, provide:
- Id (reuse the existing ids where possible)
- Name
- Plural name
- One-line definition focused on applicability and how is it different from other conditions.
- Guidelines (5-8 bullets) for extracting instances/values of the condition — concise rules for deriving further instances/values from data (e.g., what signals to look for, what is the format of the instances/values, what are the meaningful/canonical distinctions between instances/values, etc.). Keep bullets short and actionable.
- Example instances/values (1-2) — representative and illustrative only, not an exhaustive list.

Here is the list of pieces of information:
```
{pieces}```

Here is the list of candidate conditions:
```
{candidates}```
"""


SCHEMA_FORMAT = """[{schema_id}] {schema_title} (Plural: {schema_plural})
Definition: {schema_definition}
Condition Guidelines:
{schema_guidelines}
Example condition cases: 
{schema_labels}"""

def candidates_to_str(candidates):
    candidates_str = ""
    for candidate_idx, candidate in enumerate(candidates):
        guidelines_str = ""
        for guideline in candidate["codebook_guidelines"]:
            guidelines_str += f"\t- {guideline}\n"
        if len(candidate["codebook_guidelines"]) == 0:
            guidelines_str = "No guidelines for defining instances of this condition.\n"
        labels_str = ""
        for label in candidate["labels"][:3]:
            labels_str += f"\t- {label['title']}: {label['definition']}\n"

        if len(candidate["labels"]) == 0:
            labels_str = "No example instances of this condition (i.e., labels).\n"
        schema_str = SCHEMA_FORMAT.format(schema_id=f"F{candidate_idx+1}", schema_title=candidate["schema"], schema_plural=candidate["schema_plural"], schema_definition=candidate["definition"], schema_guidelines=guidelines_str, schema_labels=labels_str)
        candidates_str += f"{schema_str}\n"
    return candidates_str

def candidates_gen_to_struct(gen_candidates):
    struct_candidates = []
    for candidate in gen_candidates:
        labels = []
        for label in candidate["examples"]:
            labels.append({
                "title": label["title"],
                "definition": label["definition"],
                "examples": [],
            })
        struct_candidates.append({
            "schema": candidate["title"],
            "schema_plural": candidate["title_plural"],
            "definition": candidate["definition"],
            "codebook_guidelines": candidate["guidelines"],
            "labels": labels,
        })
    return struct_candidates


def form_facet_candidates(task, pieces, current_candidates):
    pieces_str = ""
    for piece_idx, piece in enumerate(pieces):
        pieces_str += f"[{piece_idx+1}] {piece['content']}\n"

    candidates_str = candidates_to_str(current_candidates)
    
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_FORM_FACET_CANDIDATES,
        },
        {
            "role": "user",
            "content": USER_PROMPT_FORM_FACET_CANDIDATES.format(task=task, pieces=pieces_str, candidates=candidates_str),
        },
    ]
    response = get_response_pydantic(messages, CandidateConditionsSchema)

    merged_candidates = combine_facet_candidates(
        task,
        current_candidates,
        candidates_gen_to_struct(response["candidate_conditions"]),
    )

    return merged_candidates


SYSTEM_PROMPT_COMBINE_FACET_CANDIDATES = """
You are a helpful assistant who can understand and analyze procedural knowledge and its applicability."""

USER_PROMPT_COMBINE_FACET_CANDIDATES = """
You are given two lists of candidate conditions that can describe the scope of applicability of pieces of knowledge about {task}: an OLD list and a NEW list of conditions.

Condition Definition: A condition is a set of instances/values that can be used to describe the scope of applicability of a piece of knowledge. For example, if there is a piece of knowledge X (e.g., instruction) that applies at a step Y, then the condition is "step". and Y is the instance of the condition "step".

Produce a SINGLE COMPREHENSIVE but ORTHOGONAL list of conditions by merging the two lists. When mergining the lists, follow these steps to make decisions on pairs of conditions:

1. If condition X encompasses condition Y, then discard Y.
2. If condition X and Y are similar, then either:
    - try merging them into a single condition and see if it still satisfies the definition of the condition and remain concise and orthogonal to other conditions.
    - replace them with two conditions A and B that together can represent X and Y, but are not similar to each other.
3. If condition X and Y are different, keep both conditions.

Return the merged list of conditions.
For each condition, provide:
- Id (reuse the existing ids where possible)
- Name
- Plural name
- One-line definition focused on applicability and how is it different from other conditions.
- Guidelines (5-8 bullets) for extracting instances/values of the condition — concise rules for deriving further instances/values from data (e.g., what signals to look for, what is the format of the instances/values, what are the meaningful/canonical distinctions between instances/values, etc.). Keep bullets short and actionable.
- Example instances/values (1-2) — representative and illustrative only, not an exhaustive list.

Here is the OLD list of conditions:
```
{old_candidates}```

Here is the NEW list of conditions:
```
{new_candidates}```
"""

def combine_facet_candidates(task, prev_candidates, new_candidates):
    prev_candidates_str = candidates_to_str(prev_candidates)
    new_candidates_str = candidates_to_str(new_candidates)

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_COMBINE_FACET_CANDIDATES,
        },
        {
            "role": "user",
            "content": USER_PROMPT_COMBINE_FACET_CANDIDATES.format(task=task, old_candidates=prev_candidates_str, new_candidates=new_candidates_str),
        },
    ]
    response = get_response_pydantic(messages, CandidateConditionsSchema)

    merged_candidates = candidates_gen_to_struct(response["candidate_conditions"])
    return merged_candidates