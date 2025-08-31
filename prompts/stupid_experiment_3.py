import json
from helpers import get_response_pydantic

from pydantic_models.experiment_3 import SegmentationSchema

from pydantic_models.experiment_3 import InformationPiecesSchema

from pydantic_models.experiment_3 import ItemListSchema

from pydantic_models.experiment_3 import LabeledPiecesSchema

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
If you identify new {schema_plural} that are not in the existing list, add them appropriately at the end of the list.

Here is the existing list of {schema_plural}:
```
{items}```

Here is the transcript with time-stamps (in seconds):
```
{transcript}```

Return a series of {schema_plural} as a list (both old and new). When providing the examples for each {schema}, try to ensure that there is a variety of examples (around 1-3) based on the given transcript or from the existing list of {schema_plural}."""

ITEM_FORMAT = """[{item_id}] {item_label}
Definition: {item_definition}
Examples: 
{examples}"""

ITEM_EXAMPLE_FORMAT = """\t- Context {example_idx}: {example_context}
\t- Content {example_idx}: {example_content}
"""


def form_context_codebook(task, transcript, schema):
    transcript_str = ""
    for i, subtitle in enumerate(transcript):
        start_str = f"{int(subtitle['start'])}"
        end_str = f"{int(subtitle['end'])}"
        transcript_str += f"[{start_str} - {end_str}] {subtitle['text']}\n"

    guidelines_str = ""
    for guideline in schema["codebook_guidelines"]:
        guidelines_str += f"- {guideline}\n"

    items_str = ""
    for item_idx, item in enumerate(schema["labels"]):
        examples_str = ""
        for example_idx, example in enumerate(item["examples"]):
            examples_str += ITEM_EXAMPLE_FORMAT.format(example_context=example["context"], example_content=example["content"], example_idx=example_idx + 1)
        item_id = f"L{item_idx + 1}"
        items_str += ITEM_FORMAT.format(item_id=item_id, item_label=item["title"], item_definition=item["definition"], examples=examples_str)
    
    items_str = f"No {schema['schema_plural']} yet. Define new ones."

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_PROCESS_TRANSCRIPT,
        },
        {
            "role": "user",
            "content": USER_PROMPT_FORM_CONTEXT_CODEBOOK.format(task=task, schema_plural=schema["schema_plural"], schema=schema["schema"], definition=schema["definition"], guidelines=guidelines_str, items=items_str, transcript=transcript_str)
        },
    ]

    response = get_response_pydantic(messages, ItemListSchema)

    items = response["items"]
    for item in items:
        del item["id"]
    return items


USER_PROMPT_LABEL_TRANSCRIPT_PIECES = """
You are analyzing a tutorial video for {task}.
You are given a list of pieces of information from a tutorial (recipe, SOP, repair guide, etc.) along with a list of possible {schema_plural} involved in the task. A {schema} is {definition}.

Your task is to read through the pieces of information sequentially and label the pieces of information with the appropriate {schema}.
The {schema_plural} may not be in order in the list, and some {schema_plural} may not be used at all. Only assign a {schema} when the content clearly matches the {schema}. If it does not match any {schema}, leave it as empty string `""`.

Here is the list of {schema_plural}:
```
{items}```

Here is the list of pieces of information with ids in square brackets `[]` (e.g., `[piece_id] content`):
```
{pieces}```

Return the pieces of information with labels in the same order as they were provided."""

def label_transcript_pieces(task, pieces, schema):
    pieces_str = ""
    for piece_idx, piece in enumerate(pieces):
        pieces_str += f"[{piece_idx+1}] {piece['content']}\n"

    items_str = ""
    for item_idx, item in enumerate(schema["labels"]):
        examples_str = ""
        for example_idx, example in enumerate(item["examples"]):
            examples_str += ITEM_EXAMPLE_FORMAT.format(example_context=example["context"], example_content=example["content"], example_idx=example_idx + 1)
        item_id = f"L{item_idx + 1}"
        items_str += ITEM_FORMAT.format(item_id=item_id, item_label=item["title"], item_definition=item["definition"], examples=examples_str)
    if items_str == "":
        print("STRONG WARNING: No items found in the schema.")
        return []

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_PROCESS_TRANSCRIPT,
        },
        {
            "role": "user",
            "content": USER_PROMPT_LABEL_TRANSCRIPT_PIECES.format(task=task, pieces=pieces_str, schema_plural=schema["schema_plural"], schema=schema["schema"], definition=schema["definition"], items=items_str),
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




