from pydantic_models.framework import InformationPiecesSchema

from pydantic_models.framework import create_labeled_pieces_schema

from prompts import transcript_to_str, pieces_to_str, vocabulary_to_str, guidelines_to_str

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


SYSTEM_PROMPT_EXTRACT_PIECES_FROM_TRANSCRIPT = """
You are a helpful assistant who can understand and analyze knowledge about {task} from tutorial videos."""

USER_PROMPT_EXTRACT_PIECES_FROM_TRANSCRIPT_COARSE = """
From a tutorial-style transcript (recipe, SOP, repair guide, etc.) extract all information pieces relevant to the task. Follow the procedure below.

### PROCEDURE
1. Detect task-relevant information
- Parse the transcript clause-by-clause.
- Create one piece for each indivisible action, object, reason, or tip.
    - Indivisible = removing any word breaks the meaning.
- If a sentence contains multiple actions (`whisk, then fold`) or multiple rationales, split them.
- Rewrite each piece so it is understandable standing alone—no dangling `it`,`they`, `this step`, etc.
Examples:
- `Whisk the eggs for 30 s, then fold in the flour.` -> `Whisk the eggs for 30 seconds.` `Fold the eggs in the flour.`
- `Let the paint dry so it won't smudge.` -> `Let the paint dry.` (method) `Drying the paint prevents smudging.` (explanation)
- `Add the beaten eggs to the mixture and mix well.` -> `Add the beaten eggs to the mixture.` `Mix beaten eggs well.`
- `Add 1 cup of white chocolate chips and stir until thoroughly combined.` -> `Add 1 cup of white chocolate chips.` `Stir white chocolate chips until thoroughly combined.`

2. Assign `content_type` to each piece according to below taxonomy:
<type>
    <title> Greeting </title>
    <definition> Opening and closing remarks. </definition>
    <subtypes>
        <subtype>
            <title> Greeting - Opening </title>
            <definition> Starting remarks and instructor/channel introductions. </definition>
            <example> Hey, what's up you guys, Chef [...] here. </example>
        </subtype>
        <subtype>
            <title> Greeting - Closing </title>
            <definition> Parting remarks and wrap-up. </definition>
            <example> Stay tuned, we'll catch you all later. </example>
        </subtype>
    </subtypes>
</type>
<type>
    <title> Overview </title>
    <definition> Main purpose of the video and its descriptions. </definition>
    <subtypes>
        <subtype>
            <title> Overview - Goal </title>
            <definition> Main purpose of the video and its descriptions. </definition>
            <example> Today, I'll show you a special technique which is totally special and about image pressing. </example>
        </subtype>
        <subtype>
            <title> Overview - Motivation </title>
            <definition> Reasons or background information on why the video was created. </definition>
            <example> [...] Someone is making a very special valentine's day meal for another certain special someone. </example>
        </subtype>
        <subtype>
            <title> Overview - Briefing </title>
            <definition> Rundown of how the goal will be achieved. </definition>
            <example> I'm pretty sure that just taking a pencil and putting it over the front and then putting a bunch of rubber bands around the pencil [...] that's going to do it. </example>
        </subtype>
    </subtypes>
</type>
<type>
    <title> Method </title>
    <definition> Actions that the instructor performs to complete the task. </definition>
    <subtypes>
        <subtype>
            <title> Method - Subgoal </title>
            <definition> Objective of a subsection. </definition>
            <example> Now for the intricate layer that will give me the final webbing look. </example>
        </subtype>
        <subtype>
            <title> Method - Instruction </title>
            <definition> Actions that the instructor performs to complete the task. </definition>
            <example> We're going to pour that into our silicone baking cups. </example>
        </subtype>
        <subtype>
            <title> Method - Tool </title>
            <definition> Introduction of the materials, ingredients, and equipment to be used. </definition>
            <example> I'm also going to use a pair of scissors, a glue stick, some fancy tape or some regular tape. </example>
        </subtype>
    </subtypes>
</type>
<type>
    <title> Supplementary </title>
    <definition> Additional instructions or information that makes instructions easier, faster, or more efficient. </definition>
    <subtypes>
        <subtype>
            <title> Supplementary - Tip </title>
            <definition> Additional instructions or information that makes instructions easier, faster, or more efficient. </definition>
            <example> I find that it's easier to do just a couple of layers at a time instead of all four layers at a time. </example>
        </subtype>

        <subtype>
            <title> Supplementary - Warning </title>
            <definition> Actions that should be avoided. </definition>
            <example> I don't know but I would say avoid using bleach if you can. </example>
        </subtype>
    </subtypes>
</type>
<type>
    <title> Explanation </title>
    <definition> Reasons why the instruction was performed. </definition>
    <subtypes>
        <subtype>
            <title> Explanation - Justification </title>
            <definition> Reasons why the instruction was performed. </definition>
            <example> Because every time we wear our contact lenses, makeup and even dirt particles [...] might harm our eyes directly. </example>
        </subtype>
        <subtype>
            <title> Explanation - Effect </title>
            <definition> Consequences of the instruction. </definition>
            <example> And these will overhang a little to help hide the gap. </example>
        </subtype>
    </subtypes>
</type>
<type>
    <title> Description </title>
    <definition> Descriptions of the current state of the target object. </definition>
    <subtypes>
        <subtype>
            <title> Description - Status </title>
            <definition> Descriptions of the current state of the target object. </definition>
            <example> Something sticky and dirty all through the back seat. </example>
        </subtype>
        <subtype>
            <title> Description - Context </title>
            <definition> Descriptions of the method or the setting. </definition>
            <example> [...] The process of putting on a tip by hand [...] takes a lot of patience but it can be done if you're in a pinch. </example>
        </subtype>
        <subtype>
            <title> Description - Tool Specification </title>
            <definition> Descriptions of the tools and equipment. </definition>
            <example> These are awesome beans, creamy texture, slightly nutty loaded with flavor. </example>
        </subtype>
    </subtypes>
</type>
<type>
    <title> Conclusion </title>
    <definition> Descriptions of the final results of the procedure. </definition>
    <subtypes>
        <subtype> 
            <title> Conclusion - Outcome </title>
            <definition> Descriptions of the final results of the procedure. </definition>
            <example> And now we have a dinosaur taggy blanket that wrinkles, so a fun gift for any baby on your gift giving list. </example>
        </subtype>
        <subtype>
            <title> Conclusion - Reflection </title>
            <definition> Summary, evaluation, and suggestions for the future about the overall procedure. </definition>
            <example> However, I am still concerned about how safe rubbing alcohol actually is to use so maybe next time, I will give vodka a try. </example>
        </subtype>
    </subtypes>
</type>
<type>
    <title> Miscellaneous </title>
    <definition> Personal stories, jokes, user engagement, and advertisements. </definition>
    <subtypes>
        <subtype>
            <title> Miscellaneous - Side Note </title>
            <definition> Personal stories, jokes, user engagement, and advertisements. </definition>
            <example> Tristan is back from basketball - He made it on the team so it's pretty exciting. </example>
        </subtype>
        <subtype>
            <title> Miscellaneous - Self-promotion </title>
            <definition> Promotion of the instructor of the channel (i.e. likes, subscription, notification, or donations). </definition>
            <example> So if you like this video, please give it a thumbs up and remember to subscribe. </example>
        </subtype>
        <subtype>
            <title> Miscellaneous - Bridge </title>
            <definition> Meaningless phrases or expressions that connect different sections. </definition>
            <example> And we're going to go ahead and get started. </example>
        </subtype>
        <subtype>
            <title> Miscellaneous - Filler </title>
            <definition> Conventional filler words. </definition>
            <example> Whoops. </example>
        </subtype>
    </subtypes>
</type>

3. Time-stamps
- For video/audio transcripts, copy the exact start and end (in seconds) that bracket the source text of the item.
- If no timing metadata exists, output null for both.

### INPUTS
- The transcript with time-stamps (in seconds) is as follows:
{transcript}

### OUTPUT
Return a list of information pieces, classified based on the taxonomy above."""

def extract_pieces_from_transcript_request(task, transcript, extraction_model, **kwargs):

    transcript_str = transcript_to_str(transcript)

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_EXTRACT_PIECES_FROM_TRANSCRIPT.format(task=task),
        },
        {
            "role": "user",
            "content": USER_PROMPT_EXTRACT_PIECES_FROM_TRANSCRIPT_COARSE.format(transcript=transcript_str),
        },
    ]
    return {
        "messages": messages,
        "response_format": InformationPiecesSchema,
        "model": extraction_model,
    }

def extract_pieces_from_transcript_response(response, **kwargs):
    return response["pieces"]


SYSTEM_PROMPT_LABEL_TRANSCRIPT_PIECES = """
You are a helpful assistant who can perform temporal segmentation of a tutorial-style transcript about {task}."""

USER_PROMPT_LABEL_TRANSCRIPT_PIECES = """
Given pieces of information from a tutorial about the task, perform temporal segmentation based on {facet_plural} (i.e., {definition}). Assign exactly one segment label from the provided vocabulary. If no segment label clearly apply, assign the "na" label. Follow the procedure below.

### PROCEDURE
For each piece, assign exactly one segment label from the provided vocabulary following segmentation guidelines:
{guidelines}
- If multiple segment labels seem plausible, pick the single best segment label that is most specific to the information piece's content.
- Do not use external knowledge. Ground decisions strictly in the provided transcript.
- If no segment label clearly applies, assign the "na" label.
- Make sure not to drop or merge pieces.

### INPUTS
- Canonical segment labels ([ID] `label` -- `definition`):
{vocabulary}
- Pieces of information ([ID] `content`):
{pieces}

### OUTPUT
Return a labeled list of pieces in the original order."""

def label_transcript_pieces_request(task, pieces, facet, generation_model, **kwargs):
    pieces_str = pieces_to_str(pieces)
    vocabulary_str = vocabulary_to_str(facet["vocabulary"])
    guidelines_str = guidelines_to_str(facet["guidelines"])
    facet_plural = facet["title_plural"]
    definition = facet["definition"]
    
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_LABEL_TRANSCRIPT_PIECES.format(task=task),
        },
        {
            "role": "user",
            "content": USER_PROMPT_LABEL_TRANSCRIPT_PIECES.format(pieces=pieces_str, facet_plural=facet_plural, definition=definition, vocabulary=vocabulary_str, guidelines=guidelines_str),
        },
    ]
    label_choices = [label["label"] for label in facet["vocabulary"]]
    response_format = create_labeled_pieces_schema(
        label_choices=label_choices,
        min_length=len(pieces),
    )

    return {
        "messages": messages,
        "response_format": response_format,
        "model": generation_model,
    }

def label_transcript_pieces_response(facet, pieces, response, **kwargs):
    possible_labels = set([label["label"].strip().lower() for label in facet["vocabulary"]])
    labeled_pieces = response["labeled_pieces"]
    piece_to_label = {}
    for labeled_piece in labeled_pieces:
        idx = int(labeled_piece["piece_id"]) - 1
        if idx < 0 or idx >= len(pieces):
            print(f"WARNING: Invalid piece ID: {labeled_piece['piece_id']}")
            continue
        cur_piece_id = pieces[idx]["piece_id"]
        if cur_piece_id in piece_to_label:
            ### Reasoning: A particular segmentation may suggest multiple labels for the same piece, but we enforce only one label per piece to simplify the labeling process (disallowing multi-label classification). For consistency, we assign the first suggested label and ignore the rest. If the labels beyond the first are important for discriminating the piece (to reach the desired level of discriminativeness), the pipeline is designed to be able to accommodate this by proposing additional segmentation dimension (i.e., aspects of task context).
            print(f"WARNING: Multiple labels found for the same piece: {cur_piece_id}")
            continue
        clean_label = labeled_piece["label"].strip().lower()
        # if '[' in clean_label and ']' in clean_label:
        #     clean_label = clean_label.split(']')[1].strip().lower()
        if clean_label not in possible_labels:
            print(f"WARNING: Invalid label found for the piece: {cur_piece_id} {clean_label}")
            continue
        piece_to_label[cur_piece_id] = clean_label

    formatted_pieces = []
    for piece in pieces:
        cur_piece_id = piece["piece_id"]
        if cur_piece_id in piece_to_label:
            formatted_pieces.append(piece_to_label[cur_piece_id])
        else:
            formatted_pieces.append("na")
            ### Reasoning: If no label is found for the piece, we assign the "na" label by default to indicate that the piece cannot be segmented based on the current segmentation.
            print(f"WARNING: No label found for the piece ID: {cur_piece_id}")

    return formatted_pieces