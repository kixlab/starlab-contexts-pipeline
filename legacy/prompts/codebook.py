
from pydantic_models.framework import VocabularySchema, SegmentationFacetSchema
from prompts import transcript_to_str, vocabulary_to_str, guidelines_to_str, segmentation_candidates_gen_to_struct

SYSTEM_PROMPT_FORM_CONTEXT_CODEBOOK = """
You are discovering segment labels for temporal segmentation of a tutorial about {task}."""

USER_PROMPT_FORM_CONTEXT_CODEBOOK = """
Given a transcript, find a list of segment labels for temporal segmentation of the tutorial-style transcript based on {facet_plural} (i.e., {definition}). Follow the procedure below.

### PROCEDURE
1. Based on the segmentation guidelines, identify the segment labels that can be used to segment the given transcript. Use concise, unambiguous phrasing; avoid brand-specific or overly narrow wording unless essential to task fidelity.
2. Ensure that the segment labels are semantically distinct (i.e., do not overlap with respect to the definition).
3. Ensure that the segment labels cover the entire transcript (i.e., all the pieces can be assigned a segment label).

### INPUTS
- Example segment labels:
{vocabulary}
- Tutorial-style transcript with timestamps (in seconds):
{transcript}    

### GUIDELINES FOR SEGMENTATION:
{guidelines}

### OUTPUT
Return a list of segment labels for temporal segmentation of the transcript based on {facet_plural}."""

def form_codebook_request(task, transcript, facet, generation_model, **kwargs):
    transcript_str = transcript_to_str(transcript)
    guidelines_str = guidelines_to_str(facet["guidelines"])
    vocabulary_str = vocabulary_to_str(facet["vocabulary"])

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_FORM_CONTEXT_CODEBOOK.format(task=task),
        },
        {
            "role": "user",
            "content": USER_PROMPT_FORM_CONTEXT_CODEBOOK.format(facet_plural=facet["title_plural"], definition=facet["definition"], guidelines=guidelines_str, transcript=transcript_str, vocabulary=vocabulary_str)
        },
    ]
    return {
        "messages": messages,
        "response_format": VocabularySchema,
        "model": generation_model,
    }

def form_codebook_response(response, **kwargs):
    vocabulary = response["vocabulary"]
    for label in vocabulary:
        del label["id"]
        label["label"] = label["label"].strip().lower()
    return vocabulary

SYSTEM_PROMPT_COMBINE_CODEBOOKS = """
You are fine-tuning the segmentation definition, guidelines, and vocabulary for temporal segmentation of a tutorial on {task}."""

USER_PROMPT_COMBINE_CODEBOOKS = """
Given different variations of the segmentation vocabulary (i.e., segment labels) for temporal segmentation of a tutorial-style transcript based on {facet_plural}, synthesize a unified, comprehensive segmentation (i.e., its definition, guidelines, and vocabulary) that combines them all, but preserves the nuances of each variant as much as possible. Remember that, the segmentation is defined as a temporal segmentation axis of the tutorial-style transcript that segments the transcript into meaningful segments. Its segmentation guidelines should ensure that the segmentation is non-overlapping, covers the entire transcript, and each segment must be labeled with only one segment label (i.e., no multi-label classification). Follow the procedure below.

### PROCEDURE
1. Synthesize a single set of vocabulary that combines all the variants. Combine similar labels, but try to preserve the nuances as much as possible. The final set of labels (i.e., the vocabulary) should cover all the labels in all the variants, but have no redundant labels.
    - If a label is similar to another label, combine them into a single label.
    - If a label is a more general version of another label, remove the more general label and keep the more specific label.
    - Ensure that the vocabulary is comprehensive (i.e., it covers all the labels in all the variants).
    - Ensure that no label is repeated in the final vocabulary.
2. Fine-tune the segmentation definition to exactly to fit the new vocabulary.
3. Fine-tune the segmentation guidelines to exactly to fit the new vocabulary.

### INPUTS
- Different variations of the segmentation vocabulary (i.e., segment labels):
{vocabularies}
- Initial segmentation definition:
{definition}
- Initial segmentation guidelines:
{guidelines}

### OUTPUT
Return an updated segmentation definition, guidelines, and vocabulary for temporal segmentation of the tutorial-style transcript based on {facet_plural} that combines all the variations."""

def combine_codebooks_request(task, facet, vocabularies, generation_model, **kwargs):
    guidelines_str = guidelines_to_str(facet["guidelines"])
    vocabulary_strs = []
    for vocabulary in vocabularies:
        vocabulary_strs.append(vocabulary_to_str(vocabulary))
    
    vocabularies_str = ("\n" + "-"*100 + "\n").join(vocabulary_strs)

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_FORM_CONTEXT_CODEBOOK.format(task=task),
        },
        {
            "role": "user",
            "content": USER_PROMPT_COMBINE_CODEBOOKS.format(facet_plural=facet["title_plural"], definition=facet["definition"], guidelines=guidelines_str, vocabularies=vocabularies_str),
        }
    ]
    return {
        "messages": messages,
        "response_format": SegmentationFacetSchema,
        "model": generation_model,
    }

def combine_codebooks_response(response, **kwargs):
    return segmentation_candidates_gen_to_struct([response])[0]

SYSTEM_PROMPT_TRY_ADDING_NA_LABEL = """
You are helpful assistant who can analyze a list of labels for temporal segmentation of a tutorial about {task}."""

USER_PROMPT_TRY_ADDING_NA_LABEL = """
Given a list of labels for temporal segmentation of tutorials, ensure that there is a clear `na` (i.e., "not applicable") label by either updating an existing label or proposing a new one. Follow the procedure below.

### PROCEDURE
1. Analyze the labels and determine if any one of the labels can be used to label the cases where no segment label applies (i.e., `na`). If no label can be used, propose a new `na` label.
2. In either case, make sure that the `na` label is clear, easy to understand, and has the exact label "na".

### INPUTS
{vocabulary}

### OUTPUT
Return an updated list of labels that includes the `na` label."""
### Reasoning: explicitly adding the "na" label is necessary to avoid inconsistent labeling of `does not apply` cases (i.e., when some cases are labeled as `does not apply` (i.e, ""), some are labeled as "unspecified" (i.e, unspecified is discovered as part of the vocabulary).
def try_adding_na_label_request(task, vocabulary, generation_model, **kwargs):
    vocabulary_str = vocabulary_to_str(vocabulary)
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_TRY_ADDING_NA_LABEL.format(task=task),
        },
        {
            "role": "user",
            "content": USER_PROMPT_TRY_ADDING_NA_LABEL.format(vocabulary=vocabulary_str),
        },
    ]

    return {
        "messages": messages,
        "response_format": VocabularySchema,
        "model": generation_model,
    }

def try_adding_na_label_response(response, **kwargs):
    vocabulary = response["vocabulary"]
    for label in vocabulary:
        del label["id"]
        label["label"] = label["label"].strip().lower()
    return vocabulary
