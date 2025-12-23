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

from prompts.label import label_transcript_pieces_request, label_transcript_pieces_response, extract_pieces_from_transcript_request, extract_pieces_from_transcript_response
from prompts.codebook import form_codebook_request, form_codebook_response, combine_codebooks_request, combine_codebooks_response, try_adding_na_label_request, try_adding_na_label_response

from prompts.segmentation_facet import form_segmentation_facet_candidates_request, form_segmentation_facet_candidates_response, combine_segmentation_facet_candidates_request, combine_segmentation_facet_candidates_response
from prompts.segmentation_facet_v2 import form_segmentation_facet_candidates_request_v2, form_segmentation_facet_candidates_response_v2
from prompts.framework_batch import batch_run_lm_calls

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
            cell_id = f"<Information Type: {piece['content_type']}>"
            for facet in context_schema:
                key = facet["id"]
                cell_id += f"<{facet['title']}: {get_label(piece, key)}>"
            ### add the piece contexts
            cell_to_units[cell_id][piece["unit_id"]].append(piece)
            relevant_units.add(piece["unit_id"])
    
    return cell_to_units, len(relevant_units)

def extract_pieces(task, dataset=None, context_length=None, extraction_model=None):
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
    
    if dataset is None:
        raise ValueError("Dataset is required for extracting pieces")
    if context_length is None:
        raise ValueError("Context length is required for extracting pieces")
    if extraction_model is None:
        raise ValueError("Extraction model is required for extracting pieces")

    request_args = []
    req_idx_to_source = []
    for idx, video in enumerate(dataset):
        ### forming the information units (conceptually should be easily redefinable)
        request_args.append({
            "task": task,
            "transcript": video['transcript'],
            "extraction_model": extraction_model,
        })
        req_idx_to_source.append(idx)

    batch_results = batch_run_lm_calls(request_args, extract_pieces_from_transcript_request, extract_pieces_from_transcript_response)

    for idx, pieces in zip(req_idx_to_source, batch_results):
        dataset[idx]['pieces'] = []
        for i, piece in enumerate(pieces):
            dataset[idx]['pieces'].append({
                "piece_id": f"piece_{idx}_{i}",
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

def form_information_units(dataset, information_unit_similarity_threshold, embedding_method):
    ### Reasoning: We re-cluster the information units (only assign new unit ids). This may change the shape of the dataset.
    all_pieces = []
    for video in dataset:
        for piece in video['pieces']:
            all_pieces.append(piece)

    #### cluster similar pieces in `all_pieces`
    unit_labels = clustering_custom([piece["content"] for piece in all_pieces], information_unit_similarity_threshold, embedding_method=embedding_method)
    for i, piece in enumerate(all_pieces):
        cur_unit_id = f"unit_{unit_labels[i]}"
        piece["unit_id"] = cur_unit_id
    
    return dataset

def build_information_units(task, dataset, context_length, information_unit_similarity_threshold, embedding_method, extraction_model):
    
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
    dataset = form_information_units(dataset, information_unit_similarity_threshold, embedding_method)

    # Save clustered result specific to the similarity threshold
    with open(clustered_path, "w") as f:
        json.dump(dataset, f, indent=4)

    return dataset

def select_biggest_cells(cell_to_units, include_cells):
    chosen_sets = defaultdict(list)
    for cell_id in cell_to_units.keys():
        if len(cell_to_units[cell_id]) <= 1:
            continue
        combined_pieces = []
        for unit_id in cell_to_units[cell_id]:
            if len(cell_to_units[cell_id][unit_id]) > 0:
                ### Reasoning: Add only one representative per unit to avoid tyring to discriminate between the same unit.
                combined_pieces.append(cell_to_units[cell_id][unit_id][0])

        chosen_sets[cell_id] = combined_pieces ## store the list of combined pieces for each cell

    if include_cells == 0 or len(chosen_sets) == 0:
        print("LOG: No new candidates will be added")
        return []

    chosen_sets = sorted(chosen_sets.items(), key=lambda x: len(x[1]), reverse=True)
    chosen_sets = chosen_sets[:include_cells]
    chosen_sets = {cell_id: combined_pieces for cell_id, combined_pieces in chosen_sets} ## convert to a dictionary
    return chosen_sets

def update_facet_candidates(task, old_facet_candidates, cell_to_units, include_cells, embedding_method, pieces_at_once, generation_model):
    """
    Update the facet candidates
    """

    chosen_sets = select_biggest_cells(cell_to_units, include_cells)
    print(f"Selected {len(chosen_sets)} cell-piece sets: {[(cell_id, len(combined_pieces)) for cell_id, combined_pieces in chosen_sets.items()]}")
    
    new_chosen_sets = defaultdict(list)
    for cell_id, combined_pieces in chosen_sets.items():
        piece_texts = [piece["content"].lower() for piece in combined_pieces]
        additional_labels = [piece["unit_id"] for piece in combined_pieces]
        distant_indices = find_most_distant_items(piece_texts, count=pieces_at_once, embedding_method=embedding_method, additional_labels=additional_labels)

        chosen_pieces = [combined_pieces[i] for i in distant_indices]

        if len(chosen_pieces) < 2:
            continue
        
        new_chosen_sets[cell_id] = chosen_pieces

    chosen_sets = new_chosen_sets

    if len(chosen_sets) == 0:
        return []

    new_facet_candidates = []

    request_args = []
    req_idx_to_source = []

    for i, (cell_id, chosen_pieces) in enumerate(chosen_sets.items()):
        request_args.append({
            "task": task,
            "pieces": chosen_pieces,
            "generation_model": generation_model,
            "old_facet_assignment": cell_id,
            "old_facet_candidates": old_facet_candidates,
        })
        req_idx_to_source.append(i)

    batch_results = batch_run_lm_calls(request_args, form_segmentation_facet_candidates_request, form_segmentation_facet_candidates_response)

    for i, retrieved_candidates in zip(req_idx_to_source, batch_results):
        new_facet_candidates.extend(retrieved_candidates)

    if len(chosen_sets) > 1 and len(new_facet_candidates) > 1:
        ### i.e., there are potentially overlapping candidates
        request_args = []
        request_args.append({
            "task": task,
            "candidates": new_facet_candidates,
            "generation_model": generation_model,
        })
        req_idx_to_source = [0]
        
        batch_results = batch_run_lm_calls(request_args, combine_segmentation_facet_candidates_request, combine_segmentation_facet_candidates_response)

        new_facet_candidates = batch_results[0]

    ### assign unique ids
    for facet in new_facet_candidates:
        facet["id"] = f"F{random_uid()}"

    print(f"Found {len(new_facet_candidates)} new facet candidates: {[(candidate['title'], candidate['id']) for candidate in new_facet_candidates]}")

    return new_facet_candidates

def update_facet_candidates_v2(task, labeled_dataset, cell_to_units, include_cells, embedding_method, pieces_at_once, generation_model):
    """
    Update the facet candidates V2
    """
    ### choose top-2 tutorials that have the most pieces in the same cell.
    if include_cells == 0: 
        return []

    tutorials = []
    target_piece_ids = []
    target_cell_id = None
    for cell_id, units in cell_to_units.items():
        unit_ids = list(units.keys())
        if (len(unit_ids) < 2):
            # can only breakdown if there are at least 2 units in the cell
            continue
        tutorial_unit_piece = defaultdict(dict)
        for tutorial in labeled_dataset:
            tutorial_id = tutorial["url"]
            for piece in tutorial["pieces"]:
                if piece["unit_id"] not in unit_ids:
                    continue
                if piece["unit_id"] not in tutorial_unit_piece[tutorial_id]:
                    tutorial_unit_piece[tutorial_id][piece["unit_id"]] = piece["piece_id"]
        ### try finding only one
        cur_tutorials = []
        cur_target_piece_ids = []
        for tutorial in labeled_dataset:
            tutorial_id = tutorial["url"]
            if len(tutorial_unit_piece[tutorial_id].keys()) > len(cur_target_piece_ids):
                cur_target_piece_ids = list(tutorial_unit_piece[tutorial_id].values())
                cur_tutorials = [tutorial]
        if len(cur_target_piece_ids) == 1:
            ### try finding two that cover the most units
            for tutorial1 in labeled_dataset:
                tutorial1_id = tutorial1["url"]
                for tutorial2 in labeled_dataset:
                    tutorial2_id = tutorial2["url"]
                    if tutorial1_id == tutorial2_id:
                        continue
                    combined_unit_ids = set(tutorial_unit_piece[tutorial1_id].keys()) | set(tutorial_unit_piece[tutorial2_id].keys())
                    if len(combined_unit_ids) > len(cur_target_piece_ids):
                        temp_target_piece_ids = []
                        for unit_id in combined_unit_ids:
                            if unit_id in tutorial_unit_piece[tutorial1_id]:
                                temp_target_piece_ids.append(tutorial_unit_piece[tutorial1_id][unit_id])
                            elif unit_id in tutorial_unit_piece[tutorial2_id]:
                                temp_target_piece_ids.append(tutorial_unit_piece[tutorial2_id][unit_id])
                        if len(temp_target_piece_ids) > len(cur_target_piece_ids):
                            cur_target_piece_ids = temp_target_piece_ids
                            cur_tutorials = [tutorial1, tutorial2]

        if len(cur_target_piece_ids) == 1:
            continue
        if len(cur_target_piece_ids) > len(target_piece_ids):
            target_piece_ids = cur_target_piece_ids
            tutorials = cur_tutorials
            target_cell_id = cell_id

    if len(target_piece_ids) < 2:
        # can only breakdown if we can cover at least 2 units
        return []

    print(f"Selected cell {target_cell_id} for breakdown: {len(target_piece_ids)} pieces: {target_piece_ids}")

    new_facet_candidates = []
    request_args = []

    request_args.append({
        "task": task,
        "tutorials": tutorials,
        "target_piece_ids": target_piece_ids,
        "generation_model": generation_model,
    })

    batch_results = batch_run_lm_calls(request_args, form_segmentation_facet_candidates_request_v2, form_segmentation_facet_candidates_response_v2)

    new_facet_candidates.extend(batch_results[0])

    ### assign unique ids
    for facet in new_facet_candidates:
        facet["id"] = f"F{random_uid()}"

    print(f"Found {len(new_facet_candidates)} new facet candidates: {[(candidate['title'], candidate['id']) for candidate in new_facet_candidates]}")

    return new_facet_candidates

def update_facet_labels(task, labeled_dataset, facet_candidates, vocabulary_iterations, generation_model):
    """
    Extract the vocabulary from each tutorial and merge.
    """

    if len(labeled_dataset) < vocabulary_iterations:
        selected_indices = range(len(labeled_dataset))
    else:
        selected_indices = random.sample(range(len(labeled_dataset)), vocabulary_iterations)

    ### extract the vocabulary for each facet
    request_args = []
    req_idx_to_source = []
    for fi, facet in enumerate(facet_candidates):
        for vi, video in enumerate([labeled_dataset[i] for i in selected_indices]):
            if len(video["pieces"]) == 0:
                continue
            request_args.append({
                "task": task,
                "transcript": video["transcript"],
                "facet": facet,
                "generation_model": generation_model,
            })
            req_idx_to_source.append(fi)
    
    batch_results = batch_run_lm_calls(request_args, form_codebook_request, form_codebook_response)

    vocabularies_per_facet = defaultdict(list)
    for fi, vocabulary in zip(req_idx_to_source, batch_results):
        key = facet_candidates[fi]["id"]
        vocabularies_per_facet[key].append(vocabulary)

    #### TODO: remove later
    for key, vocabularies in vocabularies_per_facet.items():
        print(f"LOG: Vocabulary for facet {key}:")
        print([len(vocabulary) for vocabulary in vocabularies])

    ### combine the vocabularies for each facet
    request_args = []
    req_idx_to_source = []
    for fi, facet in enumerate(facet_candidates):
        key = facet["id"]
        if len(vocabularies_per_facet[key]) == 0:
            continue
        request_args.append({
            "task": task,
            "facet": facet,
            "vocabularies": vocabularies_per_facet[key],
            "generation_model": generation_model,
        })
        req_idx_to_source.append(fi)

    batch_results = batch_run_lm_calls(request_args, combine_codebooks_request, combine_codebooks_response)

    for fi, new_facet in zip(req_idx_to_source, batch_results):
        facet_candidates[fi] = {
            **facet_candidates[fi],
            **new_facet,
        }

    #### TODO: remove later
    for facet in facet_candidates:
        print(f"LOG: Facet {facet['id']}:")
        print(facet["id"], len(facet["vocabulary"]))
        print()

    ### try adding na label for each facet
    request_args = []
    req_idx_to_source = []
    for fi, facet in enumerate(facet_candidates):
        request_args.append({
            "task": task,
            "vocabulary": facet["vocabulary"],
            "generation_model": generation_model,
        })
        req_idx_to_source.append(fi)

    batch_results = batch_run_lm_calls(request_args, try_adding_na_label_request, try_adding_na_label_response)

    for fi, vocabulary in zip(req_idx_to_source, batch_results):
        facet_candidates[fi]["vocabulary"] = vocabulary
    return facet_candidates

def update_labeled_dataset(task, labeled_dataset, facet_candidates, generation_model):
    """
    label the dataset based on the codebooks
    TODO: label multiple times per video
    TODO: may need to update `facet_candidates` if we see same facet again
    """

    request_args = []
    req_idx_to_source = []
    ### label according to candidates
    for fi, facet in enumerate(facet_candidates):
        key = facet["id"]
        if len(facet["vocabulary"]) == 0:
            continue
        for vi, video in enumerate(labeled_dataset):
            if len(video["pieces"]) == 0:
                continue
            request_args.append({
                "task": task,
                "pieces": video["pieces"],
                "facet": facet,
                "generation_model": generation_model,
            })
            req_idx_to_source.append((fi, vi))
    
    batch_results = batch_run_lm_calls(request_args, label_transcript_pieces_request, label_transcript_pieces_response)

    for (fi, vi), labeled_pieces in zip(req_idx_to_source, batch_results):
        key = facet_candidates[fi]["id"]
        if len(labeled_pieces) != len(labeled_dataset[vi]['pieces']):
            raise ValueError(f"{len(labeled_pieces)} != {len(labeled_dataset[vi]['pieces'])}")
        for piece_idx, piece in enumerate(labeled_dataset[vi]["pieces"]):
            if key not in piece["labels"]:
                piece["labels"][key] = []
            else:
                raise ValueError(f"Found the same facet again: {facet_candidates[fi]['title']} ({key})")
            piece["labels"][key].append(labeled_pieces[piece_idx])
    
    return labeled_dataset