import json
from helpers import get_response_pydantic

from pydantic_models.framework import InformationPiecesSchema

from pydantic_models.framework import VocabularySchema

from pydantic_models.framework import LabeledPiecesSchema

from pydantic_models.framework import CandidateSegmentationFacetsSchema

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


SYSTEM_PROMPT_ANALYSIS = """
You are a helpful assistant who can understand and analyze tutorial videos about {task}."""

USER_PROMPT_EXTRACT_PIECES_FROM_TRANSCRIPT_COARSE = """
From a tutorial-style transcript (recipe, SOP, repair guide, etc.) extract all task-relevant information. Work strictly in the order below.

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

2. Assign `type` and `subtype` to each piece according to below taxonomy:
<type>
    <title> Greeting </title>
    <definition> Opening and closing remarks. </definition>
    <subtypes>
        <subtype>
            <title> Opening </title>
            <definition> Starting remarks and instructor/channel introductions. </definition>
            <example> Hey, what's up you guys, Chef [...] here. </example>
        </subtype>
        <subtype>
            <title> Closing </title>
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
            <title> Goal </title>
            <definition> Main purpose of the video and its descriptions. </definition>
            <example> Today, I'll show you a special technique which is totally special and about image pressing. </example>
        </subtype>
        <subtype>
            <title> Motivation </title>
            <definition> Reasons or background information on why the video was created. </definition>
            <example> [...] Someone is making a very special valentine's day meal for another certain special someone. </example>
        </subtype>
        <subtype>
            <title> Briefing </title>
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
            <title> Subgoal </title>
            <definition> Objective of a subsection. </definition>
            <example> Now for the intricate layer that will give me the final webbing look. </example>
        </subtype>
        <subtype>
            <title> Instruction </title>
            <definition> Actions that the instructor performs to complete the task. </definition>
            <example> We're going to pour that into our silicone baking cups. </example>
        </subtype>
        <subtype>
            <title> Tool </title>
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
            <title> Tip </title>
            <definition> Additional instructions or information that makes instructions easier, faster, or more efficient. </definition>
            <example> I find that it's easier to do just a couple of layers at a time instead of all four layers at a time. </example>
        </subtype>

        <subtype>
            <title> Warning </title>
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
            <title> Justification </title>
            <definition> Reasons why the instruction was performed. </definition>
            <example> Because every time we wear our contact lenses, makeup and even dirt particles [...] might harm our eyes directly. </example>
        </subtype>
        <subtype>
            <title> Effect </title>
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
            <title> Status </title>
            <definition> Descriptions of the current state of the target object. </definition>
            <example> Something sticky and dirty all through the back seat. </example>
        </subtype>
        <subtype>
            <title> Context </title>
            <definition> Descriptions of the method or the setting. </definition>
            <example> [...] The process of putting on a tip by hand [...] takes a lot of patience but it can be done if you're in a pinch. </example>
        </subtype>
        <subtype>
            <title> Tool Specification </title>
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
            <title> Outcome </title>
            <definition> Descriptions of the final results of the procedure. </definition>
            <example> And now we have a dinosaur taggy blanket that wrinkles, so a fun gift for any baby on your gift giving list. </example>
        </subtype>
        <subtype>
            <title> Reflection </title>
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
            <title> Side Note </title>
            <definition> Personal stories, jokes, user engagement, and advertisements. </definition>
            <example> Tristan is back from basketball - He made it on the team so it's pretty exciting. </example>
        </subtype>
        <subtype>
            <title> Self-promotion </title>
            <definition> Promotion of the instructor of the channel (i.e. likes, subscription, notification, or donations). </definition>
            <example> So if you like this video, please give it a thumbs up and remember to subscribe. </example>
        </subtype>
        <subtype>
            <title> Bridge </title>
            <definition> Meaningless phrases or expressions that connect different sections. </definition>
            <example> And we're going to go ahead and get started. </example>
        </subtype>
        <subtype>
            <title> Filler </title>
            <definition> Conventional filler words. </definition>
            <example> Whoops. </example>
        </subtype>
    </subtypes>
</type>

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

def extract_pieces_from_transcript(task, transcript, extraction_model):

    transcript_str = transcript_to_str(transcript)

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_ANALYSIS.format(task=task),
        },
        {
            "role": "user",
            "content": USER_PROMPT_EXTRACT_PIECES_FROM_TRANSCRIPT_COARSE.format(transcript=transcript_str),
        },
    ]

    response = get_response_pydantic(messages, InformationPiecesSchema, model=extraction_model)

    pieces = response["pieces"]
    return pieces

USER_PROMPT_FORM_CONTEXT_CODEBOOK = """
You are discovering segment labels for temporal segmentation of a tutorial based on {facet_plural} (i.e., {definition}).

### INPUTS
- Existing segment labels:
{vocabulary}
- Tutorial-style transcript with timestamps (in seconds):
{transcript}    

### GOAL
Return an updated list of segment labels for temporal segmentation of the tutorial based on {facet_plural} (including existing and any new ones needed to cover the given transcript), each with 1-2 concrete examples grounded in the given transcript (or reused from the existing labels).

### GUIDELINES FOR SEGMENTATION:
{guidelines}

### GENERAL NORMALIZATION RULES (apply in addition to the above):
- Merge synonyms/variants into one segment label; keep the existing labels as the canonical form when applicable.
- Use concise, unambiguous phrasing; avoid brand-specific or overly narrow wording unless essential to task fidelity.
- Only add a new segment label if the transcript contains a segment that is not covered by any existing segment label.

### PROCEDURE
1) Segment the transcript temporally following the segmentation guidelines.
2) For each segment:
    - Try to map it to an existing segment label.
    - If no existing segment label fits, propose a new segment label following the guidelines.
3) For every segment label in the final list (existing or new):
    - Provide 1-2 representative examples.
    - Each example must include:
        - content (the exact minimal excerpt that would be labeled as the segment label)
        - context (around 10-20 words surrounding the content and the content itself)
    - If a segment label is not evidenced in the given transcript, reuse one example from the existing list if available.

### OUTPUT
Return a list of segment labels. Each segment label should have the following fields:
- id: id of the segment label
- label: short text less than 2-3 words
- definition: elaboration of what the segment label means
- examples: 1-2 short representative content and context that would be labeled as the segment label

NOTES
- Ground everything in the provided inputs; no external knowledge or speculation.
- Do not create duplicate or overlapping segment labels.
- Keep examples short and specific; avoid paraphrasing for the 'content' field."""

LABEL_FORMAT = """[{label_id}] {label}
Definition: {definition}
Examples (Contents and Context): ```
{examples}
```"""

LABEL_EXAMPLE_FORMAT = """\t- Content {example_idx}: {example_content}
\t- Context {example_idx}: {example_context}"""

def examples_to_str(examples):
    if len(examples) == 0:
        return "No examples contents and context are available."
    
    example_strs = []
    for example_idx, example in enumerate(examples):
        example_strs.append(LABEL_EXAMPLE_FORMAT.format(example_context=example["context"], example_content=example["content"], example_idx=example_idx + 1))
    example_str = "\n".join(example_strs)
    return example_str

def vocabulary_to_str(vocabulary):
    if len(vocabulary) == 0:
        return "No labels are available."

    labels_strs = []
    for label_idx, label in enumerate(vocabulary):
        label_id = f"S{label_idx + 1}"
        labels_strs.append(LABEL_FORMAT.format(label_id=label_id, label=label["label"].strip().lower(), definition=label["definition"], examples=examples_to_str(label["examples"])))

    vocabulary_str = "\n".join(labels_strs)
    return vocabulary_str

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

    guidelines_str = guidelines_to_str(facet["guidelines"])

    vocabulary_str = vocabulary_to_str(facet["vocabulary"])

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_ANALYSIS.format(task=task),
        },
        {
            "role": "user",
            "content": USER_PROMPT_FORM_CONTEXT_CODEBOOK.format(facet_plural=facet["title_plural"], facet=facet["title"], 
            definition=facet["definition"], guidelines=guidelines_str, vocabulary=vocabulary_str, transcript=transcript_str)
        },
    ]

    response = get_response_pydantic(messages, VocabularySchema)

    vocabulary = response["vocabulary"]
    for label in vocabulary:
        del label["id"]
        label["label"] = label["label"].strip().lower()
    return vocabulary

USER_PROMPT_TRY_ADDING_NA_LABEL = """
You are given a list of segment labels. Ensure that there is a clear `NA` label by either updating an existing label or proposing a new one.

### INPUTS
{vocabulary}

### PROCEDURE
1. Analyze the labels and determine if any one of the labels can be used to label the case where no segment label applies (i.e., `NA`). If no label can be used, propose a new `NA` label.
2. In either case, make sure that the `NA` label is clear, easy, and has the exact label "na".
"""
### Reasoning: explicitly adding the "na" label is necessary to avoid inconsistent labeling of `does not apply` cases (i.e., when some cases are labeled as `does not apply` (i.e, ""), some are labeled as "unspecified" (i.e, unspecified is discovered as part of the vocabulary).
def try_adding_na_label(task, vocabulary):
    iterations = 0
    while iterations < 10:
        vocabulary_str = vocabulary_to_str(vocabulary)
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT_ANALYSIS.format(task=task),
            },
            {
                "role": "user",
                "content": USER_PROMPT_TRY_ADDING_NA_LABEL.format(vocabulary=vocabulary_str),
            },
        ]
        response = get_response_pydantic(messages, VocabularySchema)
        vocabulary = response["vocabulary"]
        for label in vocabulary:
            del label["id"]
            label["label"] = label["label"].strip().lower()
        ### check if there is a label that is exactly "NA"
        for label in vocabulary:
            if label["label"] == "na":
                return vocabulary
    raise ValueError("No exact \"na\" label was added to the vocabulary after 10 iterations.")

USER_PROMPT_LABEL_TRANSCRIPT_PIECES = """
You are performing temporal segmentation of a tutorial based on {facet_plural} (i.e., {definition}).

### INPUTS
- Canonical segment labels (each starts with an ID in square brackets, e.g., "[S1] ..."):
{vocabulary}
- Pieces of information (each starts with an ID in square brackets, e.g., "[p12] ..."):
{pieces}

### GOAL
For each piece, assign exactly one segment label from the provided labels.
If no segment label clearly apply, assign the "na" label.

### MATCHING RULES
- Choose only from the provided segment labels; output the segment label exactly as given.
- If multiple segment labels seem plausible, pick the single best segment label that is most specific to the information piece's content.
- Do not use external knowledge. Ground decisions strictly in the provided transcript.

### OUTPUT
Return a list of labeled pieces. Each piece should have the following fields:
- piece_id: id of the piece
- label_id: id of the segment label
- label: segment label or "na"

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

    vocabulary_str = vocabulary_to_str(facet["vocabulary"])
    
    if len(facet["vocabulary"]) == 0:
        print("STRONG WARNING: No labels found for the facet.")
        return []

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_ANALYSIS.format(task=task),
        },
        {
            "role": "user",
            "content": USER_PROMPT_LABEL_TRANSCRIPT_PIECES.format(pieces=pieces_str, facet_plural=facet["title_plural"], definition=facet["definition"], vocabulary=vocabulary_str),
        },
    ]
    response = get_response_pydantic(messages, LabeledPiecesSchema)

    labeled_pieces = response["labeled_pieces"]
    piece_to_label = {}
    for labeled_piece in labeled_pieces:
        cur_piece_id = pieces[int(labeled_piece["piece_id"])-1]["piece_id"]
        if cur_piece_id in piece_to_label:
            print("STRONG WARNING: Multiple labels found for the same piece ID.", cur_piece_id)
        clean_label = labeled_piece["label"].strip().lower()
        if '[' in clean_label and ']' in clean_label:
            clean_label = clean_label.split(']')[1].strip().lower()
        piece_to_label[cur_piece_id] = clean_label

    formatted_pieces = []
    for piece in pieces:
        cur_piece_id = piece["piece_id"]
        if cur_piece_id in piece_to_label:
            formatted_pieces.append(piece_to_label[cur_piece_id])
        else:
            print("STRONG WARNING: No label found for the piece ID.", cur_piece_id)
            formatted_pieces.append("")

    return formatted_pieces




SYSTEM_PROMPT_DESCRIBE_CONTEXTS = """You are a helpful assistant who can reliably and precisely describe task contexts where pieces of information apply."""

USER_PROMPT_FORM_SEGMENTATION_FACET_CANDIDATES = """
You are given information pieces from tutorial videos about {task}. Identify a set of aspects of a task context (a particular temporal segmentation) that would assign DIFFERENT segment labels to the pieces.

### INPUTS
{pieces}

### POSSIBLE TYPES OF ASPECTS:
| Type | Example Titles | Example Labels | Example Distinction |
|------|----------------|----------------|----------------------|
| When | Stage of process | Setup / Execution | Different steps in time |
| Why | Purpose / Subgoal | Collect Data / Analyze Data | Different goals or intentions |
| Where | Environment | Field / Lab | Different physical or digital settings |
| What | Object of focus | Hardware / Software | Working on different components |
| How | Method / Tool used | Manual / Automated | Using different approaches or tools |

### PROCEDURE
1. Identify at least one aspect of a task context that would assign DIFFERENT segment labels to the pieces. You can list multiple aspects if you can, but make sure they are orthogonal (i.e., do not overlap with respect to type).
2. Classify which type of aspect it belongs to ("when", "why", "where", "what", or "how").
3. Briefly justify the choice of the aspect and the type of segmentation.
4. Describe how the task can be segmented or divided along this aspect.  
5. Provide brief "guidelines" that explain how to identify segment boundaries or assign labels based on the transcript of a tutorial. 
6. Provide a few examples of "segment labels".

### ANNOTATION GUIDELINES
- Use only the transcript text to infer the distinction.
- Keep aspect titles short, but easily interpretable (max. 2-3 words).
- Keep example segment labels short, but easily interpretable (max. 2-3 words).
"""

CANDIDATE_FACET_FORMAT = """[{id}] ({type}) {title} (Plural: {title_plural}) -- {definition}
Guidelines for defining labels: ```
{guidelines}
```
Example labels: ```
{vocabulary}
```"""

PIECE_FORMAT = """Information {idx}: {content}
Surrounding context (tutorial excerpt): ```{context}```"""

def segmentation_candidates_gen_to_struct(gen_candidates):
    struct_candidates = []
    for candidate in gen_candidates:
        segment_labels = []
        for label in candidate["segment_labels"]:
            segment_labels.append({
                "label": label["label"],
                "definition": label["definition"],
                "examples": [],
            })
        struct_candidates.append({
            "type": candidate["type"],
            "title": candidate["aspect"],
            "title_plural": candidate["aspect_plural"],
            "definition": candidate["segmentation"],
            "guidelines": candidate["segmentation_guidelines"],
            "vocabulary": segment_labels,
        })
    return struct_candidates

def candidates_to_str(candidates):
    candidates_strs = []
    for idx, candidate in enumerate(candidates):
        guidelines_str = guidelines_to_str(candidate["guidelines"])
        vocabulary_str = vocabulary_to_str(candidate["vocabulary"])
        candidates_strs.append(CANDIDATE_FACET_FORMAT.format(
            id=f"F{idx+1}",
            type=candidate["type"],
            title=candidate["title"],
            title_plural=candidate["title_plural"],
            definition=candidate["definition"],
            guidelines=guidelines_str,
            vocabulary=vocabulary_str,
        ))
    candidates_str = "\n".join(candidates_strs)
    return candidates_str

def form_segmentation_facet_candidates(task, pieces):

    pieces_str = ""
    for piece_idx,piece in enumerate(pieces):
        pieces_str += PIECE_FORMAT.format(idx=piece_idx+1, content=piece["content"], context=piece["raw_context"]) + "\n"

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_DESCRIBE_CONTEXTS,
        },
        {
            "role": "user",
            "content": USER_PROMPT_FORM_SEGMENTATION_FACET_CANDIDATES.format(task=task, 
            pieces=pieces_str),
        },
    ]
    response = get_response_pydantic(messages, CandidateSegmentationFacetsSchema)

    # merged_candidates = combine_facet_candidates(
    #     task,
    #     current_candidates,
    #     candidates_gen_to_struct(response["candidates"]),
    # )

    # return merged_candidates
    return segmentation_candidates_gen_to_struct(response["candidates"])

USER_PROMPT_COMBINE_SEGMENTATION_FACET_CANDIDATES = """
You are analyzing aspects of a task context (a particular temporal segmentation) for `{task}`/

### INPUTS
- Aspects of a task context, each labeled with an ID in square brackets (e.g., "[F1] ..."): {candidates}

### GOAL
Combine the given aspects of a task context to get a set of aspects of a task context that are orthogonal (the segmentation wrt one aspect does not determine the segmentation wrt another aspect), single-slot (the segmentation title and labels are short and easily interpretable), and cover all initial aspects of a task context.

### POSSIBLE TYPES OF ASPECTS:
| Type | Example Titles | Example Labels | Example Distinction |
|------|----------------|----------------|----------------------|
| When | Stage of process | Setup / Execution | Different steps in time |
| Why | Purpose / Subgoal | Collect Data / Analyze Data | Different goals or intentions |
| Where | Environment | Field / Lab | Different physical or digital settings |
| What | Object of focus | Hardware / Software | Working on different components |
| How | Method / Tool used | Manual / Automated | Using different approaches or tools |

### PROCEDURE
Iterate until stable:
1) Check for orthogonality:
    - Redefine or replace non-orthogonal pairs with independent aspects of a task context.  
    - Example (WRONG: not independent):  
        - [F3] Physical settings
        - [F4] Settings
        - Settings are more general than physical settings, so can focus on one of the two.
    - Example (CORRECT: independent redesign):  
        - [F4] Settings
2) Check for single-slot:
    - Rephrase aspects of a task context so that segmentation title and labels fit in max. 3 words, but are easily interpretable.
    - Example (WRONG: too long):  
        - [F5] Task phase segmentation
    - Example (CORRECT: single-slot):  
        - [F5] Phases
4) Check for coverage:
    - Ensure the new set preserves all the meaning of the original aspects of a task context.

### EXIT CONDITION
The final aspect set is smallest (i.e., has the fewest number of aspects), orthogonal, single-slot, and fully covers the initial aspects.
"""

def combine_segmentation_facet_candidates(task, all_candidates):
    all_candidates_str = candidates_to_str(all_candidates)

    messages = [

        {
            "role": "system",
            "content": SYSTEM_PROMPT_DESCRIBE_CONTEXTS,
        },
        {
            "role": "user",
            "content": USER_PROMPT_COMBINE_SEGMENTATION_FACET_CANDIDATES.format(task=task, candidates=all_candidates_str),
        },
    ]
    response = get_response_pydantic(messages, CandidateSegmentationFacetsSchema)

    return segmentation_candidates_gen_to_struct(response["candidates"])