import json
from helpers import get_response_pydantic

from pydantic_models.experiment_3 import SegmentationSchema

from pydantic_models.experiment_3 import InformationPiecesSchema

from pydantic_models.experiment_3 import LabelListSchema

from pydantic_models.experiment_3 import LabeledPiecesSchema

from pydantic_models.experiment_3 import CandidateApplicabilityFacetsSchema

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
You are given a list of pieces of information from a tutorial-style transcript (recipe, SOP, repair guide, etc.) along with a list of possible {schema_plural} involved in the task. A {schema} is {definition}.

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


SYSTEM_PROMPT_FORM_FACET_CANDIDATES = """You are a helpful assistant who identifies and refines `applicability facets` that define why/when/where pieces of information about a procedural task apply.
"""

### TODO: try directly asking why/when/where the pieces of information apply?
USER_PROMPT_FORM_FACET_CANDIDATES = """
You are analyzing pieces of information from different tutorial-style transcripts (recipe, SOP, repair guide, etc.) for {task}.

INPUTS:
- Information items:
{pieces}
- Candidate APPLICABILITY FACETS:
{candidates}

GOAL:
Update the existing list of APPLICABILITY FACETS to differentiate given information items (i.e., each semantically distinct item can be given a unique applicability signature across the chosen FACETS). Reuse/extend existing FACETS first; add new ones only if required to resolve remaining collisions.

DEFINITIONS:
- APPLICABILITY FACET: one atomic discriminator on a single axis type (Why | When | Where).
- FACET VALUES: mutually exclusive options for that discriminator. In this task, give only 1-2 illustrative examples per FACET (not exhaustive).
- Atomicity rule: reject umbrella FACETS (e.g., context/method); split into minimal, non-overlapping discriminators.

TYPES OF APPLICABILITY FACETS:
- Why: the outcome/intent the information item advances.
- When: the procedural stage or a state (incl. phase/step, preconditions, version/time window) that the information item applies to.
- Where: the environment/setting in which the information item holds.

PROCEDURE: repeat steps 1-6 until no collisions (no indistinguishable pairs) and no redundant facets remain.
1) Find collisions: identify semantically different item pairs that remain indistinguishable under the current facet set; note the concrete discriminator(s) hinted by the items.
2) REUSE: map each discriminator to an existing facet. If it fits, keep the facet.
3) UPDATE (make atomic and clear): if a reused facet is umbrella/multi-axis, split or tighten it so each facet encodes exactly one discriminator on one axis and still covers old+new meaning.
4) ADD (only if needed): if a collision persists and no existing facet can host the discriminator, introduce one new atomic facet of a one of the three types (Why | When | Where).
5) MERGE: merge facets that encode the same discriminator, ensuring that they are atomic and clear; If there are irrelevant facets, keep them for comprehensiveness.
exit condition: all item pairs are separated by at least one facet.

OUTPUT: An updated list of APPLICABILITY FACETS
For each APPLICABILITY FACET, provide:
- Id (reuse if possible)
- Type: Why | When | Where
- Name (<=4 words)
- Definition (<= 20 words)
- Guidelines (5-8 bullets) to define and extract the FACET VALUES (signals to look for, the formats of the values, etc.)
- Example FACET VALUES (1-2): Value name — value definition

NOTES:
- Keep APPLICABILITY FACET definitions concrete;
- FACET VALUES are illustrative only; do not attempt completeness.
"""


SCHEMA_FORMAT = """[{schema_id}] (Type: {schema_type}) {schema_title} (Plural: {schema_plural})
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
        schema_str = SCHEMA_FORMAT.format(schema_id=f"F{candidate_idx+1}", schema_type=candidate["schema_type"], schema_title=candidate["schema"], schema_plural=candidate["schema_plural"], schema_definition=candidate["definition"], schema_guidelines=guidelines_str, schema_labels=labels_str)
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
            "schema_type": candidate["type"],
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
    if len(current_candidates) == 0:
        candidates_str = "No candidates provided."
    
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
    response = get_response_pydantic(messages, CandidateApplicabilityFacetsSchema)

    # merged_candidates = combine_facet_candidates(
    #     task,
    #     current_candidates,
    #     candidates_gen_to_struct(response["candidates"]),
    # )

    # return merged_candidates
    return candidates_gen_to_struct(response["candidates"])


# SYSTEM_PROMPT_COMBINE_FACET_CANDIDATES = """
# You are a helpful assistant who can understand and analyze procedural knowledge and its applicability."""

# USER_PROMPT_COMBINE_FACET_CANDIDATES = """
# You are given two lists of candidate CONDITIONS that describe where knowledge about {task} applies:
# - OLD candidates:
# {old_candidates}
# - NEW candidates:
# {new_candidates}

# GOAL
# Produce ONE comprehensive, orthogonal set of atomic CONDITIONS (with concrete illustrative example VALUES). Prefer NEW conditions only if coverage is equal or better.

# DEFINITION
# - CONDITION = one atomic discriminator on a single axis: Purpose (why/outcome) | Method (how/technique) | Context (state/equipment/constraints, incl. workflow phase/step).
# - VALUES = mutually-exclusive options for that discriminator.
# - Reject umbrella categories (e.g., "context/strategy/handling")—split into atomic CONDITIONS.

# REPRESENTATIVE EXAMPLES (for guidance only)
# - Purpose example — Removal intent
#     Values: Seamless realism, Privacy redaction, Make layout space.
# - Method example — Removal technique
#     Values: Clone/heal brush, Content-aware/patch, Generative inpainting.
# - Context example — Removal scale
#     Values: Spot/blemish, Small object, Large/structural.

# PROCESS
# 1) Combine: Simply concatenate OLD + NEW into one working list.
# 2) Remove duplicates and subset conditions: 
#     - Duplicate: If two conditions have the same axis and definition.
#     - Subset: If one of the conditions completely contain the other.
#    Positive examples — MERGE:
#    - Context: "Removal scale" vs "Object size"
#      • Same discriminator (area/px). → Keep "Removal scale".
#    - Method: "Inpainting method" vs "Synthesis method"
#      • Same discriminator (how pixels are generated). → Keep "Synthesis method".
#    - Purpose: "Seamless realism" vs "Invisible cleanup"
#      • Same discriminator (goal: invisibility). → Keep "Seamless realism" (NEW wording).

#    Negative examples — DO NOT MERGE:
#    - Cross-axis: "Removal intent" (Purpose) vs "Clone/Heal" (Method)
#      • Different axes → keep both.
#    - Same axis, different discriminator: "Background variation" vs "Lighting condition" (Context)
#      • Different ideas (texture vs illumination) → keep both.
#    - Not a subset: "Convection bake" (device) vs "High-then-lower" (temperature profile) (Context)
#      • Independent discriminators → keep both.

# 3) Non-orthogonal pairs (split or reframe)
#    Overlap test:
#    - Q1: Could the **same instance** satisfy both conditions *for the same reason*?
#    - Q2: Do both conditions name the **same discriminator**?
#    If YES to both → they collide; **split/reframe** so each names one atomic discriminator with non-overlapping applicability tests.

#    Positive examples — SPLIT / REFRAME:
#    - Mixed axes in one label: "Quick fix for small objects"
#      • Mixes Method (quick fix) + Context (small objects).
#      → Split into "Removal technique" (Method) and "Removal scale" (Context).
#    - Same axis collision: "Background complexity" vs "Texture continuity" (Context)
#      • Both encode variation of background; overlap on signals ("busy", "patterned").
#      → Reframe into a single "Background variation" with values {Uniform, Repetitive pattern, Multi-texture}.
#    - Purpose/Context tangle: "Seamless realism" (Purpose) vs "Invisibility level" (Context with strength buckets)
#      • Goal vs quality threshold collide in wording.
#      → Keep Purpose "Seamless realism"; move numeric thresholds under Context "Obfuscation strength"; remove duplicate purpose value.

#    Negative examples — DO NOT SPLIT:
#    - Co-applicable but different discriminators (same axis): "Synthesis method" (Method) and "Selection method" (Method)
#      • An edit rightly has both a selection and a synthesis; no collision → keep both.
#    - Different axes by design: "Privacy redaction" (Purpose) and "Pixelate" (Method)
#      • Goal vs tool; co-apply is intended → keep both.
#    - Values already mutually exclusive: "Removal scale" {Spot, Small, Large}
#      • No overlap between values; no split needed.

# 4) Loop
#    - If you merged or split in Steps 2–3, **repeat Step 2** on the updated list until no more merges/splits trigger.

# 5) Coverage check
#    - Map every OLD/NEW item → exactly one final CONDITION (record the mapping).
#    - If something doesn’t map, add a new atomic CONDITION (1 axis, 1 discriminator, clear signals).

# OUTPUT — New list of relevant CONDITIONS
# For each CONDITION, provide:
# - Id (reuse if possible)
# - Axis: Purpose | Method | Context
# - Name (<=4 words)
# - Definition (<= 20 words)
# - Guidelines (5-8 bullets) to extract VALUES (signals, pattern/threshold formats, canonical distinctions, synonym handling, when to split/merge)
# - Example VALUES (1-2): {Value name — value definition}
# """

# def combine_facet_candidates(task, prev_candidates, new_candidates):
#     prev_candidates_str = candidates_to_str(prev_candidates)
#     new_candidates_str = candidates_to_str(new_candidates)

#     messages = [
#         {
#             "role": "system",
#             "content": SYSTEM_PROMPT_COMBINE_FACET_CANDIDATES,
#         },
#         {
#             "role": "user",
#             "content": USER_PROMPT_COMBINE_FACET_CANDIDATES.format(task=task, old_candidates=prev_candidates_str, new_candidates=new_candidates_str),
#         },
#     ]
#     response = get_response_pydantic(messages, CandidateApplicabilityFacetsSchema)

#     merged_candidates = candidates_gen_to_struct(response["candidates"])
#     return merged_candidates