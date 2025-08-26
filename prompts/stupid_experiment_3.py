import json
from helpers import get_response_pydantic

from pydantic_models.experiment_3 import SegmentationSchema, ClassificationSchema, InformationPiecesSchema, StructuredPiecesSchema, LabeledPiecesSchema

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

SYSTEM_PROMPT = """
You are a helpful assistant who can understand and analyze tutorial videos.
"""

USER_SEGMENT_TRANSCRIPT_PROMPT = """
Given a tutorial video's transcript, please segment it into atomic information pieces. Then, classify each piece into one of the following categories:
```
{taxonomy}
```

The transcript is as follows:
```
{transcript}
```

Ensure that the transcript is segmented into pieces that are atomic and do not overlap. Respond with a JSON object containing the segmentation and the category of each piece:
```
{example}
```
"""

EXAMPLE_SEGMENT_TRANSCRIPT = {
    "segments": [
        {
            "text": "...", # the text of the segment
            "type": "..." # one of the categories in the taxonomy
        },
        {
            "text": "...", # the text of the segment
            "type": "..." # one of the categories in the taxonomy
        },
    ]
}

def segment_transcript_stupid(transcript):

    def format_response(response):
        return json.dumps(response, indent=2)

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": USER_SEGMENT_TRANSCRIPT_PROMPT.format(transcript=transcript, taxonomy=TAXONOMY, example=format_response(EXAMPLE_SEGMENT_TRANSCRIPT)),
        },
    ]

    response = get_response_pydantic(messages, SegmentationSchema)
    return response


SYSTEM_PROMPT_VID2COACH = """
You are a helpful assistant who can understand and analyze tutorial videos.
Given the definitions of the taxonomy, classify the provided sentence into one of the eight categories: [Greeting, Overview, Method, Supplementary,
Explanation, Description, Conclusion, and Miscellaneous]. Do not add sub category.
1. Greeting
Opening: Starting remarks and instructor/channel introductions.
Example: "Hey, what’s up you guys, Chef [...] here."
Closing: Parting remarks and wrap-up.
Example: "Stay tuned, we’ll catch you all later."
2. Overview
Goal: Main purpose of the video and its descriptions.
Example: "Today, I’ll show you a special technique which is totally special and about image pressing."
Motivation: Reasons or background information on why the video was created.
Example: "[...] Someone is making a very special valentine’s day meal for another certain special someone."
Briefing: Rundown of how the goal will be achieved.
Example: "I’m pretty sure that just taking a pencil and putting it over the front and then putting a bunch of rubber bands around the pencil
[...] that’s going to do it."
3. Method
Subgoal: Objective of a subsection.
Example: "Now for the intricate layer that will give me the final webbing look."
Instruction: Actions that the instructor performs to complete the task.
Example: "We’re going to pour that into our silicone baking cups."
Tool: Introduction of the materials, ingredients, and equipment to be used.
Example: "I’m also going to use a pair of scissors, a glue stick, some fancy tape or some regular tape."
4. Supplementary
Tip: Additional instructions or information that makes instructions easier, faster, or more efficient.
Example: "I find that it’s easier to do just a couple of layers at a time instead of all four layers at a time."
Warning: Actions that should be avoided.
Example: "I don’t know but I would say avoid using bleach if you can."
5. Explanation
Justification: Reasons why the instruction was performed.
Example: "Because every time we wear our contact lenses, makeup and even dirt particles [...] might harm our eyes directly."
Effect: Consequences of the instruction.
Example: "And these will overhang a little to help hide the gap."
6. Description
Status: Descriptions of the current state of the target object.
Example: "Something sticky and dirty all through the back seat."
Context: Descriptions of the method or the setting.
Example: "[...] The process of putting on a tip by hand [...] takes a lot of patience but it can be done if you’re in a pinch."
Tool Specification: Descriptions of the tools and equipment.
Example: "These are awesome beans, creamy texture, slightly nutty loaded with flavor."
7. Conclusion
Outcome: Descriptions of the final results of the procedure.
Example: "And now we have a dinosaur taggy blanket that wrinkles, so a fun gift for any baby on your gift giving list."
Reflection: Summary, evaluation, and suggestions for the future about the overall procedure.
Example: "However, I am still concerned about how safe rubbing alcohol actually is to use so maybe next time, I will give vodka a try."
8. Miscellaneous
Side Note: Personal stories, jokes, user engagement, and advertisements.
Example: "Tristan is back from basketball - He made it on the team so it’s pretty exciting."
Self-promotion: Promotion of the instructor of the channel (i.e. likes, subscription, notification, or donations).
Example: "So if you like this video, please give it a thumbs up and remember to subscribe."
Bridge: Meaningless phrases or expressions that connect different sections.
Example: "And we’re going to go ahead and get started."
Filler: Conventional filler words.
Example: "Whoops."
EXAMPLES:
Sentence: Hey, I’m John Kanell.
Category: Greeting
Sentence: And today on Preppy Kitchen, we’re making some quick and delicious cranberry orange muffins.
Category: Overview
"""

def classify_sentence_vid2coach(sentence):
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": sentence,
        },
    ]

    response = get_response_pydantic(messages, ClassificationSchema)
    category = response.category
    return category


SYSTEM_PROMPT_PROCESS_TRANSCRIPT = """
You are a helpful assistant who can understand and analyze tutorial videos.
"""

USER_PROMPT_FORM_INFORMATION_UNITS_FEW_SHOT = """
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
- For video/audio transcripts, copy the exact start_time and end_time in ISO 8601 seconds ("123.45") that bracket the source text of the item.
- If no timing metadata exists, output null for both.

The transcript is as follows:
```
{transcript}
```
"""

def form_information_units_few_shot(task, transcript):

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
            "content": USER_PROMPT_FORM_INFORMATION_UNITS_FEW_SHOT.format(task=task, transcript=transcript_str),
        },
    ]

    response = get_response_pydantic(messages, InformationPiecesSchema)

    pieces = response["pieces"]
    return pieces

USER_PROMPT_ASSIGN_CONTEXT_ZERO_SHOT = """
You are analyzing a tutorial video for {task}.

From a list of atomic information pieces, derive the procedure segments and the procedural scope of each piece. Work strictly in the order below.

1. Derive canonical procedure_segments
- For every `Method` piece, convert it to <object(s)> or <action> or <action> + <object(s)>, where <action> is a single verb and <object(s)> is a single object or a list of objects. Do not include any adverbs.
- Include the task as one of the procedure_segments (e.g., `make muffins`)
Example:
- `Whisk the eggs for 30 s` → [`whisk eggs`]
- `Add the beaten eggs to the mixture` → [`add eggs`]
- `Add 1 cup of white chocolate chips.` → [`add white chocolate chips`]
- Use each of these canonical phrases as a unique procedure_segment identifier.


2. Label the procedural scope of every piece
- The `procedure_segments` field is the smallest set of identifiers that fully covers the scope of the piece.
- Usually one segment; may be a set if the statement spans multiple steps.
- If the piece itself is a `Method`, its procedure_segments should normally be the single canonical action+objects segment you just created in Step 1.
- For content types `Description`, `Explanation`, and `Supplementary`, select the procedure segments that are most relevant to the piece.
- For content types `Greeting`, `Overview`, `Conclusion`, and `Miscellaneous`, leave the field empty.
- Always reuse an existing identifier rather than creating synonyms.
- All content types except `Greeting`, `Overview`, `Conclusion`, and `Miscellaneous` should have at least one procedure segment that is relevant to the piece.

3. Validation checklist (run before returning)
a.	Every content is atomic: deleting any word breaks comprehension.
b.	Every `method` appears verb-first and has at least one object.
c.	No dangling pronouns. 
d.  Every `procedure_segment` referenced actually exists.
e.  Every piece has at least one `procedure_segment`, except for `Greeting`, `Overview`, `Conclusion`, and `Miscellaneous` pieces.

The list of atomic information pieces is as follows:
```
{pieces}
```
"""

def assign_contexts_zero_shot(task, pieces):
    pieces_str = ""

    for piece in pieces:
        start_str = f"{int(piece['start'])}"
        end_str = f"{int(piece['end'])}"
        pieces_str += f"[{start_str} - {end_str}] {piece['content']} ({piece['content_type']})\n"
    
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_PROCESS_TRANSCRIPT,
        },
        {
            "role": "user",
            "content": USER_PROMPT_ASSIGN_CONTEXT_ZERO_SHOT.format(task=task, pieces=pieces_str),
        },
    ]
    response = get_response_pydantic(messages, StructuredPiecesSchema)
    pieces = response["pieces"]
    return pieces

USER_PROMPT_ASSIGN_CONTEXT_FEW_SHOT = """
You are analyzing a tutorial video for {task}.

From a list of atomic information pieces, identify the procedural scope of each piece. Follow the steps and refer to the examples below.

Label the procedural scope of every piece of information:
- The `context_step` is the description of the step that the piece of information is about.
- For content types `Greeting`, `Overview`, `Conclusion`, and `Miscellaneous`, leave the field empty.
- Always reuse an existing identifier rather than creating synonyms/equivalen `context_steps`s.
- All content types except `Greeting`, `Overview`, `Conclusion`, and `Miscellaneous` should have at least one `context_step` that is relevant to the piece.

Here are some examples:
```
{examples}
```

The list of atomic information pieces is as follows:
```
{pieces}
```
"""

def assign_contexts_few_shot(task, pieces, examples):
    pieces_str = ""

    for piece in pieces:
        start_str = f"{int(piece['start'])}"
        end_str = f"{int(piece['end'])}"
        pieces_str += f"[{start_str} - {end_str}] {piece['content']} ({piece['content_type']})\n"

    examples_str = ""
    for example in examples:
        examples_str += f"{example['content']} ({example['content_type']}) --> {example['context_step']}\n"
    
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_PROCESS_TRANSCRIPT,
        },
        {
            "role": "user",
            "content": USER_PROMPT_ASSIGN_CONTEXT_FEW_SHOT.format(task=task, pieces=pieces_str, examples=examples_str),
        },
    ]
    response = get_response_pydantic(messages, LabeledPiecesSchema)
    pieces = response["pieces"]
    return pieces

