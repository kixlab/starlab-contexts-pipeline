from pydantic_models.framework import CandidateSegmentationFacetsSchema
from prompts import (
    segmentation_candidates_gen_to_struct,
    PIECE_FORMAT,
)

SYSTEM_PROMPT_DESCRIBE_CONTEXTS_V2 = """You are a helpful assistant who can analyze and describe the contexts where knowledge applies or does not apply for task {task}."""

USER_PROMPT_FORM_SEGMENTATION_FACET_CANDIDATES_V2 = """
Given tutorials and targeted information pieces, propose a set of task context aspects (i.e., temporal segmentation axes) that would uniquely discriminate each of the targeted information pieces. Follow the requirements and the procedure below.

### REQUIREMENTS
- Each proposed aspect should be a distinct temporal segmentation axes of the tutorial-style transcript that segments the transcript into meaningful segments. Its segmentation guidelines should ensure that the segmentation is non-overlapping, covers the entire transcript, and each segment must be labeled with only one segment label (i.e., no multi-label classification).
- It should be possible to find a unique signature for each targeted piece of information (i.e., the combination of segment labels across the proposed aspects uniquely discriminates each targeted piece of information from others).
- The aspects should be orthogonal (i.e., do not overlap semantically).
- Keep aspect titles short (2-3 words), but interpretable without additional context.
- Verify that the proposed aspects can uniquely discriminate the targeted pieces of information by segmenting the provided tutorials according to the proposed aspects.

### PROCEDURE
1. Identify task context aspects (i.e., temporal segmentation axes) that would assign DIFFERENT segment labels to the targeted pieces of information.
    - Classify each aspect into one of the possible types of aspects: "when", "why", "where", "what", "how".
    - Briefly justify how the aspect would assign different segment labels to the targeted pieces of information and the choice of the type of aspect.
    - Provide a detailed definition of the aspect.
    - Provide guidelines that explain how to segment a tutorial-style transcript according to this aspect.
    - Provide a full vocabulary of segment labels that can be used to segment the tutorial-style transcript according to this aspect.
2. Segment the provided tutorials according to the proposed aspects and assign segment labels to the pieces of information in the original order.
3. If there are multiple aspects, ensure that they are orthogonal (i.e., do not overlap semantically) and that the combination of aspects uniquely discriminates each targeted piece of information from others (i.e., each targeted piece of information receives a unique combination of segment labels across the aspects).

### INPUTS
- Tutorials with their respective information pieces:
{tutorials}
- Targeted pieces of information for each tutorial:
{target_pieces}

### OUTPUT
Return a list of task context aspects (i.e., temporal segmentation axes) that satisfy the requirements."""


def form_segmentation_facet_candidates_request_v2(task, tutorials, target_piece_ids, generation_model, **kwargs):

    tutorial_pieces_strs = []
    target_piece_strs = []
    for tidx, tutorial in enumerate(tutorials):
        piece_strs = []
        cur_target_piece_strs = []
        for pidx, piece in enumerate(tutorial["pieces"]):
            cur_str = PIECE_FORMAT.format(idx=pidx+1, content=piece['content'])
            piece_strs.append(cur_str)
            if piece["piece_id"] in target_piece_ids:
                cur_target_piece_strs.append(cur_str)
        tutorial_pieces_strs.append(f"Tutorial {tidx+1}:\n" + "\n".join(piece_strs))
        target_piece_strs.append(f"Targeted pieces of information for Tutorial {tidx+1}:\n" + "\n".join(cur_target_piece_strs))
    
    tutorials_strs = "\n\n".join(tutorial_pieces_strs)
    target_pieces_str = "\n\n".join(target_piece_strs)

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_DESCRIBE_CONTEXTS_V2.format(task=task),
        },
        {
            "role": "user",
            "content": USER_PROMPT_FORM_SEGMENTATION_FACET_CANDIDATES_V2.format(
                tutorials=tutorials_strs,
                target_pieces=target_pieces_str,
            ),
        },
    ]
    return {
        "messages": messages,
        "response_format": CandidateSegmentationFacetsSchema,
        "model": generation_model,
    }


def form_segmentation_facet_candidates_response_v2(response, tutorials, target_piece_ids, **kwargs):
    #### check if the target pieces received unique signatures
    candidates = response["candidates"]
    piece_labels = {}
    for piece_id in target_piece_ids:
        piece_labels[piece_id] = {}
    for candidate in candidates:
        c_id = candidate["id"]
        for piece_id in target_piece_ids:
            piece_labels[piece_id][c_id] = "na"
        for tutorial, segmentation in zip(tutorials, candidate["segmentations"]):
            labeled_pieces = segmentation["labeled_pieces"]
            for piece in labeled_pieces:
                piece_idx = int(piece["piece_id"]) - 1
                piece_id = tutorial["pieces"][piece_idx]["piece_id"]
                if piece_id in target_piece_ids:
                    piece_labels[piece_id][c_id] = piece["label"].lower().strip()
    signatures = {}
    for piece_id in target_piece_ids:
        signatures[piece_id] = ""
        for c_id in piece_labels[piece_id]:
            signatures[piece_id] += f"<{c_id}:{piece_labels[piece_id][c_id]}>"
    
    signature_values = list(signatures.values())
    all_signatures = len(signature_values)
    unique_signatures = len(set(signature_values))
    threshold = max(1, all_signatures/5) / all_signatures ### Reasoning: we want to discriminate at least 20% of the units in each iteration if possible.
    print(f"Effectiveness of the facets ({unique_signatures}/{all_signatures}) is {unique_signatures / all_signatures}, threshold is {threshold}")
    if unique_signatures / all_signatures < threshold:
        print(f"WARNING: The targeted pieces received non-unique signatures: {signature_values}")
        return []
    
    return segmentation_candidates_gen_to_struct(candidates)
