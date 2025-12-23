CANDIDATE_FACET_FORMAT = """[{id}] ({type}) {title} (Plural: {title_plural}) -- {definition}
Guidelines for defining labels: ```
{guidelines}```
Example labels: ```
{vocabulary}```"""

GUIDELINE_FORMAT = """- {guideline}"""

LABEL_FORMAT = """[{label_id}] `{label}` -- {definition}"""

SEPARATE_PIECE_FORMAT = """Information {idx}: {content}
Surrounding context (tutorial excerpt): ```{context}```"""
PIECE_FORMAT = """[{idx}] `{content}`"""

TRANSCRIPT_FORMAT = """[{start} - {end}] {text}"""

TUTORIAL_FORMAT = """{title}\n```{content}```"""
TUTORIAL_PIECE_FORMAT = """[{id}] `{content}` (Type: {type})"""


def vocabulary_to_str(vocabulary):
    if len(vocabulary) == 0:
        return "No labels are available."

    labels_strs = []
    for label_idx, label in enumerate(vocabulary):
        labels_strs.append(LABEL_FORMAT.format(label_id=(label_idx + 1), label=label["label"].strip().lower(), definition=label["definition"]))
    return "\n".join(labels_strs)

def guidelines_to_str(guidelines):
    if len(guidelines) == 0:
        return "No guidelines are available."
    
    guideline_strs = []
    for guideline in guidelines:
        guideline_strs.append(GUIDELINE_FORMAT.format(guideline=guideline))
    return "\n".join(guideline_strs)

def pieces_to_str(pieces):
    pieces_strs = []
    for idx, piece in enumerate(pieces):
        pieces_strs.append(PIECE_FORMAT.format(idx=idx+1, content=piece['content']))
    return "\n".join(pieces_strs)

def separate_pieces_to_str(pieces):
    pieces_strs = []
    for idx, piece in enumerate(pieces):
        pieces_strs.append(SEPARATE_PIECE_FORMAT.format(idx=idx+1, content=piece["content"], context=piece["raw_context"]))
    return "\n".join(pieces_strs)

def transcript_to_str(transcript, with_timestamps=True):
    if len(transcript) == 0:
        return "No transcript provided."
    
    transcript_strs = []
    for subtitle in transcript:
        if with_timestamps:
            start_str = f"{int(subtitle['start'])}"
            end_str = f"{int(subtitle['end'])}"
            transcript_strs.append(TRANSCRIPT_FORMAT.format(start=start_str, end=end_str, text=subtitle['text']))
        else:
            transcript_strs.append(subtitle['text'])
    return "\n".join(transcript_strs)

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
    return "\n".join(candidates_strs)

def segmentation_candidates_gen_to_struct(gen_candidates):
    struct_candidates = []
    for candidate in gen_candidates:
        segment_labels = []
        for label in candidate["segment_labels"]:
            segment_labels.append({
                "label": label["label"],
                "definition": label["definition"],
            })
        struct_candidates.append({
            "type": candidate["type"],
            "title": candidate["aspect"],
            "title_plural": candidate["aspect_plural"],
            "definition": candidate["definition"],
            "guidelines": candidate["guidelines"],
            "vocabulary": segment_labels,
        })
    return struct_candidates

def tutorial_pieces_to_str(pieces):
    pieces_strs = []
    for piece in pieces:
        pieces_strs.append(TUTORIAL_PIECE_FORMAT.format(id=piece['piece_id'], content=piece['content'], type=piece['content_type']))
    return "\n".join(pieces_strs)

def tutorials_to_str(tutorials):
    tutorials_strs = []
    for tutorial in tutorials:
        content = tutorial_pieces_to_str(tutorial['pieces'])
        tutorials_strs.append(TUTORIAL_FORMAT.format(title=tutorial['title'], content=content))
    return "\n\n".join(tutorials_strs)

def tutorial_to_str(tutorial):
    content = tutorial_pieces_to_str(tutorial['pieces'])
    return TUTORIAL_FORMAT.format(title=tutorial['title'], content=content)

def response_to_str(response):
    pieces_strs = []
    for idx, piece in enumerate(response):
        pieces_strs.append(SEPARATE_PIECE_FORMAT.format(idx=idx+1, content=piece['content'], context=piece['raw_context']))
    return "\n".join(pieces_strs)