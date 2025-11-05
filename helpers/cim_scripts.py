"""
Helper functions for the CIM framework.

{label_structure} = {
    "label": {title of the label},
    "definition": {definition of the label},
    "examples": [
        {
            "context": {text surrounding the content + content that would be labeled as the label},
            "content": {the content that would be labeled as the label},
        }
    ]
}
{facet_structure} = {
    "id": {facet id based on uuid4},
    "type": {type of the facet: `why`, `when`, `where`},
    "title": {title of the facet}, 
    "title_plural": {plural form of the title},
    "definintion": {description of the facet},
    
    "guidelines": [
        {guideline for the LLM to extract the labels from the content},
        ...
    ],
    
    ### for labeling
    "vocabulary": [
        {label_structure}
        ...
    ]
}

{labeled_dataset_structure} = {
    "url": {video url}
    "title": {task title},
    "original_title": {original video title},
    "content": {concatinated transcript},
    "transcript": [
        {
            "text": {text},
            "start": {start},
            "end": {end},
        },
        ...
    ],
    "pieces": {
        {piece_id}: {
            "piece_id": {piece_id},
            "unit_id": {unit_id},
            "content": {content of the piece},
            "content_type": {type of the content: Overview, Method, Explanation, Supplementary},
            "start": {start of the content},
            "end": {end of the content},
            "labels": {
                {facet id}: [{facet value}, ...], ### number of runs
                ...
            }
        },
        ...
    }
}

approach_1_results = {
    "information_units": {
        {unit_id}: {
            "unit_id": {unit_id},
            "content": {content of the information unit},
            "content_type": {type of the content: Overview, Method, Explanation, Supplementary},
            "instances": [{id of the piece - piece_id}, ...],
        }
    },
    "context_schema": {
        {facet_id}: {facet_structure},
        ...
    },
    "facet_candidates": {
        {facet_id}: {facet_structure},
        ...
    },
    "labeled_dataset": [
        {labeled_dataset_structure},
        ...
    ]
}
"""

import json
import os
import math
import random

from collections import defaultdict

from helpers import random_uid

from helpers.nlp import clustering_custom, find_most_distant_items
from helpers.video_scripts import get_transcript_segment

from prompts.framework import extract_pieces_from_transcript
from prompts.framework import form_codebook, label_transcript_pieces
from prompts.framework import form_segmentation_facet_candidates, combine_segmentation_facet_candidates
from prompts.framework import try_adding_na_label

FRAMEWORK_PATH = "./static/results/framework/"

def calc_explained_norm(cell_to_units, relevant_units_count, base=2):

    total = math.log(relevant_units_count, base)
    d = calc_discriminativeness(cell_to_units, relevant_units_count, base)
    return (total-d) / total


def calc_discriminativeness(cell_to_units, relevant_units_count, base=2):
    ### TODO: adjust by noise

    if relevant_units_count == 0:
        return 0

    n_c_list = []

    total_entropy = 0    
    for cell_id in cell_to_units:
        n_c = len(cell_to_units[cell_id])

        if n_c == 0:
            continue

        p_c = float(n_c) / relevant_units_count
        total_entropy += p_c * math.log(n_c, base)

        n_c_list.append(n_c)

    check_entropy = math.log(relevant_units_count, base)
    for cell_id in cell_to_units:
        n_c = len(cell_to_units[cell_id])

        if n_c == 0:
            continue

        p_c = float(n_c) / relevant_units_count
        check_entropy += p_c * math.log(p_c, base)
    
    # print("LOG: Entropy check: ")
    # print(total_entropy)
    # print(check_entropy)
    # print(relevant_units_count)
    # print(n_c_list)
    # print()

    return total_entropy

def calc_compactness(context_schema, initial_labels):
    ### TODO: check if we prefer fewer facets or not
    ### need to blend the information types into context_schema
    total_labels = initial_labels-1
    for facet in context_schema:
        has_na = False
        for label in facet["vocabulary"]:
            if label["label"] == "na":
                has_na = True
                break
        total_labels += len(facet["vocabulary"]) - (1 if has_na else 0)

    return total_labels

def relative_improvement(d1, c1, d2, c2):
    compactness_weight = 0.001
    ## TODO: adjust by noise
    o1 = d1 + c1 * compactness_weight
    o2 = d2 + c2 * compactness_weight
    return o1-o2

def macro_pruning(context_schema, labeled_dataset, piece_types, threshold):
    if len(context_schema) <= 1:
        return context_schema
    
    cell_to_units, relevant_units_count = get_cell_to_units(context_schema, labeled_dataset, piece_types)
    ### macroprune the context schema by removing the least discriminative facet
    d_best = calc_discriminativeness(cell_to_units, relevant_units_count)
    c_best = calc_compactness(context_schema, len(piece_types))
    
    o_smallest = float("inf")
    facet_to_remove = None
    for i, _ in enumerate(context_schema):
        cur_context_schema = context_schema[:i] + context_schema[i+1:]
        cur_cell_to_units, cur_relevant_units_count = get_cell_to_units(cur_context_schema, labeled_dataset, piece_types)
        d = calc_discriminativeness(cur_cell_to_units, cur_relevant_units_count)
        c = calc_compactness(cur_context_schema, len(piece_types))
        
        o = relative_improvement(d, c, d_best, c_best)
        if o_smallest > o:
            o_smallest = o
            facet_to_remove = i
    if facet_to_remove is None or o_smallest > threshold:
        return context_schema, None
    removed_facet = context_schema[facet_to_remove]
    return context_schema[:facet_to_remove] + context_schema[facet_to_remove+1:], removed_facet

def get_cell_to_units(context_schema, dataset, piece_types):
    def get_label(piece, key):
        if key not in piece["labels"]:
            return ""
        if len(piece["labels"][key]) == 0:
            return ""
        ### TODO: return the last label for now, later need to be adjusted
        return piece["labels"][key][-1]

    cell_to_units = defaultdict(lambda: defaultdict(list))
    relevant_units = set()
    for video in dataset:
        for piece in video["pieces"]:
            if piece["content_type"] not in piece_types:
                continue
            cell_id = f"<{piece['content_type']}>"
            for facet in context_schema:
                key = facet["id"]
                cell_id += f"<{get_label(piece, key)}>"
            ### add the piece contexts
            cell_to_units[cell_id][piece["unit_id"]].append(piece)
            relevant_units.add(piece["unit_id"])
    
    return cell_to_units, len(relevant_units)

def extract_pieces(task, dataset, context_length, extraction_model):
    taskname = task.replace(" ", "_").lower()
    parent_path = os.path.join(FRAMEWORK_PATH, f'{taskname}')
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)

    # Paths for split outputs
    pieces_path = os.path.join(parent_path, f"extracted_pieces.json")

    # If extracted pieces exist, load and skip extraction; otherwise, extract and save
    if os.path.exists(pieces_path):
        with open(pieces_path) as f:
            return json.load(f)
    
    for video_idx, video in enumerate(dataset):
        ### forming the information units (conceptually should be easily redefinable)
        pieces = extract_pieces_from_transcript(task, video['transcript'], extraction_model)
        video['pieces'] = []
        for i, piece in enumerate(pieces):
            video['pieces'].append({
                "piece_id": f"piece_{video_idx}_{i}",
                **piece,
                "content_type": piece["type"] + " - " + piece["subtype"],
                # "type": piece["type"],
                # "subtype": piece["subtype"],
                "labels": {},
                "raw_context": "",
            })

    for video in dataset:
        for piece in video['pieces']:
            l = max(0, piece["start"]-context_length//2)
            r = piece["end"]+context_length//2
            piece["raw_context"] = get_transcript_segment(video['transcript'], l, r)
    # Save extracted pieces snapshot if we just created it (or overwrite to keep consistent)
    with open(pieces_path, "w") as f:
        json.dump(dataset, f, indent=4)
    
    return dataset

def build_information_units_v0(task, dataset, context_length, information_unit_similarity_threshold, embedding_method, extraction_model):
    
    taskname = task.replace(" ", "_").lower()
    parent_path = os.path.join(FRAMEWORK_PATH, f'{taskname}')
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)

    sim_str = str(information_unit_similarity_threshold)[2:] ## assuming it is a float that starts with `0.`
    clustered_path = os.path.join(parent_path, f"clustered_pieces_sim_{sim_str}.json")

    # If clustered result for this threshold exists, return it directly
    if os.path.exists(clustered_path):
        with open(clustered_path) as f:
            return json.load(f)

    dataset = extract_pieces(task, dataset, context_length, extraction_model)

    ### Reasoning: We cluster all information pieces at once, even the ones that may have different types. This is because we are interested in what different roles the same information pieces may play in different contexts.
    all_pieces = []
    for video in dataset:
        for piece in video['pieces']:
            all_pieces.append(piece)

    #### cluster similar pieces in `all_pieces`
    information_units = {}

    unit_labels = clustering_custom([piece["content"] for piece in all_pieces], information_unit_similarity_threshold, embedding_method=embedding_method)
    for i, piece in enumerate(all_pieces):
        cur_unit_id = f"unit_{unit_labels[i]}"
        piece["unit_id"] = cur_unit_id
        if cur_unit_id not in information_units:
            ### first piece is the representative of the cluster (IU)
            information_units[cur_unit_id] = {
                "content": piece["content"],
                "content_type": piece["content_type"],
                "instances": [piece["piece_id"]],
            }
        else:
            information_units[cur_unit_id]["instances"].append(piece["piece_id"])

    # Save clustered result specific to the similarity threshold
    with open(clustered_path, "w") as f:
        json.dump(dataset, f, indent=4)

    return dataset

def update_facet_candidates(task, cell_to_units, include_cells, embedding_method, pieces_at_once):
    """
    Update the facet candidates and the labeled dataset.
    """

    chosen_sets_of_pieces = []
    for cell_id in cell_to_units.keys():
        if len(cell_to_units[cell_id]) <= 1:
            continue
        combined_pieces = []
        for unit_id in cell_to_units[cell_id]:
            combined_pieces.extend(cell_to_units[cell_id][unit_id])

        piece_texts = [piece["content"].lower() for piece in combined_pieces]
        additional_labels = [piece["unit_id"] for piece in combined_pieces]
        distant_indices = find_most_distant_items(piece_texts, count=pieces_at_once, embedding_method=embedding_method, additional_labels=additional_labels)

        chosen_pieces = [combined_pieces[i] for i in distant_indices]

        if len(chosen_pieces) < 2:
            continue
        chosen_sets_of_pieces.append(chosen_pieces)

    chosen_sets_of_pieces = sorted(chosen_sets_of_pieces, key=lambda x: len(x), reverse=True)

    if include_cells == 0 or len(chosen_sets_of_pieces) == 0:
        print("LOG: No new candidates will be added")
    
    chosen_sets_of_pieces = chosen_sets_of_pieces[:include_cells]

    new_facet_candidates = []

    for chosen_pieces in chosen_sets_of_pieces:
        # print(json.dumps(chosen_pieces, indent=4))
        retrieved_candidates = form_segmentation_facet_candidates(task, chosen_pieces)
        for candidate in retrieved_candidates:
            new_facet_candidates.append(candidate)

    if len(chosen_sets_of_pieces) > 1:
        ### i.e., there are potentially overlapping candidates
        new_facet_candidates = combine_segmentation_facet_candidates(task, new_facet_candidates)

    ### assign unique ids
    for facet in new_facet_candidates:
        facet["id"] = f"F{random_uid()}"

    return new_facet_candidates

def update_facet_labels(task, labeled_dataset, facet_candidates, vocabulary_iterations):
    """
    similar to VideoMix --> iteratively build the codebook
    """

    if len(labeled_dataset) < vocabulary_iterations:
        selected_indices = range(len(labeled_dataset))
    else:
        selected_indices = random.sample(range(len(labeled_dataset)), vocabulary_iterations)

    for facet in facet_candidates:
        ### iteratively build the vocabulary
        for video in [labeled_dataset[i] for i in selected_indices]:
            vocabulary = form_codebook(task, video["transcript"], facet)
            facet["vocabulary"] = vocabulary
        facet["vocabulary"] = try_adding_na_label(task, facet["vocabulary"])
    return facet_candidates

def update_labeled_dataset(task, labeled_dataset, facet_candidates):
    """
    label the dataset based on the codebooks
    TODO: label multiple times per video
    TODO: may need to update `facet_candidates` if we see same facet again
    """
    ### label according to candidates
    for facet in facet_candidates:
        key = facet["id"]
        for video in labeled_dataset:
            if len(video["pieces"]) == 0:
                continue
            
            labeled_pieces = label_transcript_pieces(task, video["pieces"], facet)

            if len(labeled_pieces) != len(video["pieces"]):
                print(f"STRONG WARNING: {len(labeled_pieces)} != {len(video['pieces'])}")
            for piece_idx, piece in enumerate(video["pieces"]):
                if key not in piece["labels"]:
                    piece["labels"][key] = []
                else:
                    print(f"STRONG WARNING: found the same facet again {facet['title']}, {key}")
                piece["labels"][key].append(labeled_pieces[piece_idx])
    return labeled_dataset