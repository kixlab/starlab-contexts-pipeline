import json
import os
from collections import defaultdict

from helpers.dataset import MUFFIN_TASK, CUSTOM_TASKS, CROSS_TASK_TASKS
from helpers.dataset import IMPORTANT_TYPES_FINE

from helpers.dataset import get_dataset

from helpers.cim_scripts import get_facet_name

FRAMEWORK_PATH = "./static/results/framework/"

structs_summary = """
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
                    {facet name}: [{facet value}, ...], ### number of runs
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
            {facet_name}: {facet_structure},
            ...
        },
        "facet_candidates": {
            {facet_name}: {facet_structure},
            ...
        },
        "labeled_dataset": [
            {labeled_dataset_structure},
            ...
        ]
    }
"""

def get_cell_to_units(context_schema, dataset, important_piece_types):
    def get_label(piece, facet_name):
        if facet_name not in piece["labels"]:
            return ""
        if len(piece["labels"][facet_name]) == 0:
            return ""
        ### TODO: return the last label for now, later need to be adjusted
        return piece["labels"][facet_name][-1]

    cell_to_units = defaultdict(lambda: defaultdict(list))
    relevant_units = set()
    for video in dataset:
        for piece in video["pieces"]:
            if piece["content_type"] not in important_piece_types:
                continue
            cell_id = f"<{piece['content_type']}>"
            for facet in context_schema:
                cell_id += f"<{get_label(piece, get_facet_name(facet))}>"
            ### add the piece contexts
            cell_to_units[cell_id][piece["unit_id"]].append(piece)
            relevant_units.add(piece["unit_id"])

    ### TODO: remove sparsity check later
    # sparsity = len(important_piece_types)
    # for facet in context_schema:
    #     sparsity *= len(facet["vocabulary"]) + 1
    # sparsity = len(cell_to_units) / sparsity * 100
    # print("Sparsity %: ", sparsity)
    
    return cell_to_units, len(relevant_units)


from helpers.nlp import clustering_custom, find_most_distant_items
from prompts.framework import extract_pieces_from_transcript
from helpers.video_scripts import get_transcript_segment

def extract_pieces(task, dataset, context_length, dummy=""):
    taskname = task.replace(" ", "_").lower()
    parent_path = os.path.join(FRAMEWORK_PATH, f'{taskname}')
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)

    # Paths for split outputs
    pieces_path = os.path.join(parent_path, f"extracted_pieces_{dummy}.json")

    # If extracted pieces exist, load and skip extraction; otherwise, extract and save
    if os.path.exists(pieces_path):
        with open(pieces_path) as f:
            return json.load(f)
    
    for video_idx, video in enumerate(dataset):
        ### forming the information units (conceptually should be easily redefinable)
        pieces = extract_pieces_from_transcript(task, video['transcript'])
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

def build_information_units_v0(task, dataset, context_length, information_unit_similarity_threshold, dummy=""):
    
    taskname = task.replace(" ", "_").lower()
    parent_path = os.path.join(FRAMEWORK_PATH, f'{taskname}')
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)

    sim_str = str(information_unit_similarity_threshold)[2:] ## assuming it is a float that starts with `0.`
    clustered_path = os.path.join(parent_path, f"clustered_pieces_sim_{sim_str}_{dummy}.json")

    # If clustered result for this threshold exists, return it directly
    if os.path.exists(clustered_path):
        with open(clustered_path) as f:
            return json.load(f)

    dataset = extract_pieces(task, dataset, context_length, dummy)

    ### Reasoning: We cluster all information pieces at once, even the ones that may have different types. This is because we are interested in what different roles the same information pieces may play in different contexts.
    all_pieces = []
    for video in dataset:
        for piece in video['pieces']:
            all_pieces.append(piece)

    #### cluster similar pieces in `all_pieces`
    information_units = {}

    unit_labels = clustering_custom([piece["content"] for piece in all_pieces], information_unit_similarity_threshold, embedding_method="openai")
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

from prompts.framework import form_codebook, label_transcript_pieces

def build_codebook_v0(task, dataset, facet):
    """
    similar to VideoMix --> iteratively build the codebook
    TODO: may need to restrict the number of videos considered
    """

    ### iteratively build the context schema
    for video in dataset:
        vocabulary = form_codebook(task, video["transcript"], facet)
        facet["vocabulary"] = vocabulary

    return facet

def label_based_on_codebook_v0(task, dataset, facet):
    """
    label the dataset based on the codebook
    TODO: label multiple times per video
    """
    
    facet_name = get_facet_name(facet)

    for video in dataset:
        if "pieces" in video:
            if len(video["pieces"]) == 0:
                continue
            if facet_name in video["pieces"][0]["labels"]:
                ### skip if already labeled across `schema`
                continue

        labeled_pieces = label_transcript_pieces(task, video["pieces"], facet)

        if len(labeled_pieces) != len(video["pieces"]):
            print(f"STRONG WARNING: {len(labeled_pieces)} != {len(video['pieces'])}")
        for piece_idx, piece in enumerate(video["pieces"]):
            if facet_name not in piece["labels"]:
                piece["labels"][facet_name] = []
            piece["labels"][facet_name].append(labeled_pieces[piece_idx])

    return dataset

from prompts.framework import form_segmentation_facet_candidates, combine_segmentation_facet_candidates

def build_facet_candidates_v0(task, cell_to_units, include_cells):
    
    chosen_sets_of_pieces = []
    for cell_id in cell_to_units.keys():
        if len(cell_to_units[cell_id]) <= 1:
            continue
        combined_pieces = []
        for unit_id in cell_to_units[cell_id]:
            combined_pieces.extend(cell_to_units[cell_id][unit_id])

        piece_texts = [piece["content"].lower() for piece in combined_pieces]
        additional_labels = [piece["unit_id"] for piece in combined_pieces]
        distant_indices = find_most_distant_items(piece_texts, count=3, embedding_method="openai", additional_labels=additional_labels)

        chosen_pieces = [combined_pieces[i] for i in distant_indices]

        if len(chosen_pieces) < 2:
            continue

        print(json.dumps(chosen_pieces, indent=4))
        chosen_sets_of_pieces.append(chosen_pieces)

    chosen_sets_of_pieces = sorted(chosen_sets_of_pieces, key=lambda x: len(x), reverse=True)

    if include_cells == 0 or len(chosen_sets_of_pieces) == 0:
        print("LOG: No new candidates will be added")
    
    chosen_sets_of_pieces = chosen_sets_of_pieces[:include_cells]

    all_candidates = []

    for chosen_pieces in chosen_sets_of_pieces:
        new_facet_candidates = form_segmentation_facet_candidates(task, chosen_pieces)
        for candidate in new_facet_candidates:
            all_candidates.append(candidate)

    if len(all_candidates) > 1:
        ### i.e., there are multiple new candidates
        all_candidates = combine_segmentation_facet_candidates(task, all_candidates)
    
    return all_candidates

import math

def calc_discriminativeness(cell_to_units, relevant_units_count):
    ### TODO: adjust by noise
    base = 2

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
    
    print("LOG: Entropy check: ")
    print(total_entropy)
    print(check_entropy)
    print(relevant_units_count)
    print(n_c_list)
    print()

    return total_entropy

def calc_compactness(context_schema, initial_labels):
    ### TODO: check if we prefer fewer facets or not
    ### need to blend the information types into context_schema
    total_labels = initial_labels-1
    for facet in context_schema:
        total_labels += len(facet["vocabulary"]) ### because there is "empty" label

    return total_labels

def relative_improvement(d1, c1, d2, c2):
    compactness_weight = 0.001
    ## TODO: adjust by noise
    o1 = d1 + c1 * compactness_weight
    o2 = d2 + c2 * compactness_weight
    return o1-o2

def macro_pruning(context_schema, labeled_dataset, important_piece_types, threshold):
    if len(context_schema) <= 1:
        return context_schema
    
    cell_to_units, relevant_units_count = get_cell_to_units(context_schema, labeled_dataset, important_piece_types)
    ### macroprune the context schema by removing the least discriminative facet
    d_best = calc_discriminativeness(cell_to_units, relevant_units_count)
    c_best = calc_compactness(context_schema, len(important_piece_types))
    
    o_smallest = float("inf")
    facet_to_remove = None
    for i, _ in enumerate(context_schema):
        cur_context_schema = context_schema[:i] + context_schema[i+1:]
        cur_cell_to_units, cur_relevant_units_count = get_cell_to_units(cur_context_schema, labeled_dataset, important_piece_types)
        d = calc_discriminativeness(cur_cell_to_units, cur_relevant_units_count)
        c = calc_compactness(cur_context_schema, len(important_piece_types))
        
        o = relative_improvement(d, c, d_best, c_best)
        if o_smallest > o:
            o_smallest = o
            facet_to_remove = i
    if facet_to_remove is None or o_smallest > threshold:
        return context_schema, None
    removed_facet = context_schema[facet_to_remove]
    return context_schema[:facet_to_remove] + context_schema[facet_to_remove+1:], removed_facet

def update_facet_candidates(
    task, cell_to_units, include_cells, dummy=""
):
    """
    Update the facet candidates and the labeled dataset.
    """
    ### TODO: remove saving later
    taskname = task.replace(" ", "_").lower()
    parent_path = os.path.join(FRAMEWORK_PATH, f'{taskname}')
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)
    
    path = os.path.join(parent_path, f"facet_candidates_{dummy}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)

    new_facet_candidates = build_facet_candidates_v0(task, cell_to_units, include_cells)

    with open(path, "w") as f:
        json.dump(new_facet_candidates, f, indent=4)

    return new_facet_candidates

def update_facet_labels(task, labeled_dataset, facet_candidates, include_cells, dummy=""):
    if include_cells == 0:
        return facet_candidates

    taskname = task.replace(" ", "_").lower()
    parent_path = os.path.join(FRAMEWORK_PATH, f'{taskname}')
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)

    path = os.path.join(parent_path, f"facet_labels_{dummy}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)

    updated_candidates = []
    for schema in facet_candidates:
        updated_candidates.append(
            build_codebook_v0(task, labeled_dataset, schema)
        )

    with open(path, "w") as f:
        json.dump(updated_candidates, f, indent=4)

    return updated_candidates

def update_labeled_dataset(task, labeled_dataset, facet_candidates, dummy=""):
    
    taskname = task.replace(" ", "_").lower()
    parent_path = os.path.join(FRAMEWORK_PATH, f'{taskname}')
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)
    
    path = os.path.join(parent_path, f"labeled_dataset_{dummy}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)

    ### label according to candidates
    for schema in facet_candidates:
        ### TODO: maybe explicitly ask to skip the unimportant types?
        labeled_dataset = label_based_on_codebook_v0(task, labeled_dataset, schema)

    with open(path, "w") as f:
        json.dump(labeled_dataset, f, indent=4)

    return labeled_dataset


def process_videos(task, dataset, important_piece_types, dummy):
    ### constants
    max_iterations = 100
    pruning_interval = 5
    pruning_threshold = 1
    max_macro_pruning_len = 2
    skip_pruning = True
    stopping_delta_threshold = 0.1
    include_cells = 10
    ### Reasoning: 0.9 --> We set a higher similarity threshold to capture most of the information diversity in the dataset and remove the noise due to phrasing differences.
    information_unit_similarity_threshold=0.9
    context_length = 30 ### in seconds

    split_dummy = "split_" + dummy

    ### build the `information units`
    labeled_dataset = build_information_units_v0(task, dataset, context_length, information_unit_similarity_threshold, split_dummy)

    ### First, discover sufficient facet candidates

    facet_candidates = []
    
    base_cell_to_units, base_relevant_units_count = get_cell_to_units(context_schema, labeled_dataset, important_piece_types)
    base_d = calc_discriminativeness(base_cell_to_units, base_relevant_units_count)
    base_c = calc_compactness(context_schema, len(important_piece_types))

    context_schema = []

    iteration_insights = []
    iterations = 0

    while True:

        iterations += 1
        if iterations > max_iterations:
            print("WARNING: Maximum number of iterations reached")
            break

        ### run pruning every `pruning_interval` iterations
        if iterations % pruning_interval == 0 and not skip_pruning:
            original_length = len(context_schema)
            while len(context_schema) > 1 and len(context_schema) > original_length - max_macro_pruning_len:
                new_context_schema = macro_pruning(
                    context_schema, labeled_dataset, important_piece_types, pruning_threshold
                )
                if len(new_context_schema) < len(context_schema):
                    context_schema = new_context_schema
                    continue
                else:
                    break   
            iteration_insights.append({
                "type": "macro_pruning",
                "iteration": iterations,
                "original_length": original_length,
                "new_length": len(context_schema),
            })

        cell_to_units, relevant_units_count = get_cell_to_units(context_schema, labeled_dataset, important_piece_types)
        # iteration_insights.append({
        #     "type": "get_cell_to_units",
        #     "iteration": iterations,
        #     "cell_to_units": cell_to_units,
        # })

        ### update the facet candidates
        facet_candidates = update_facet_candidates(task, cell_to_units, facet_candidates, include_cells, app1_dummy)

        ### update the facet labels
        facet_candidates = update_facet_labels(task, labeled_dataset, facet_candidates, include_cells, app1_dummy)
        # TODO: remove this later (to search for facets again and again)
        include_cells = 0

        labeled_dataset = update_labeled_dataset(task, labeled_dataset, facet_candidates, app1_dummy)
        
        if len(facet_candidates) == 0:
            print("WARNING: No facet candidates left")
            break    

        prev_d = calc_discriminativeness(cell_to_units, relevant_units_count)
        prev_c = calc_compactness(context_schema, len(important_piece_types))
        o_biggest = -float("inf")
        best_c = 1
        best_d = prev_d
        best_facet_toadd = None
        for i, facet_candidate in enumerate(facet_candidates):
            ### get the labeling by the context schema
            candidate_context_schema = context_schema + [facet_candidate]
            candidate_cell_to_units, candidate_relevant_units_count = get_cell_to_units(candidate_context_schema, labeled_dataset, important_piece_types)
            ### calculate the discriminative and compactness
            d = calc_discriminativeness(candidate_cell_to_units, candidate_relevant_units_count)
            c = calc_compactness(candidate_context_schema, len(important_piece_types))
            o = relative_improvement(prev_d, prev_c, d, c)
            iteration_insights.append({
                "type": "relative_improvement",
                "facet_title": facet_candidate["title"],
                "iteration": iterations,
                "prev_d": prev_d,
                "prev_c": prev_c,
                "d": d,
                "c": c,
                "improvement": o,
                "absolute_improvement": relative_improvement(base_d, base_c, d, c),
            })
            if o > o_biggest:
                o_biggest = o
                best_d = d
                best_c = c
                best_facet_toadd = i

        iteration_insights.append({
            "type": "add_facet",
            "iteration": iterations,
            "prev_d": prev_d,
            "prev_c": prev_c,
            "best_d": best_d,
            "best_c": best_c,
            "improvement": o_biggest,
            "best_facet_toadd": best_facet_toadd,
            "best_facet": (None if best_facet_toadd is None else facet_candidates[best_facet_toadd]["title"]),
        })
        if o_biggest < stopping_delta_threshold:
            print("WARNING: Adding a facet didn't improve the objective `significantly`", o_biggest, stopping_delta_threshold)
            break

        if best_facet_toadd is None:
            print("WARNING: No best facet found / Seem to have converged")
            break
        context_schema.append(facet_candidates[best_facet_toadd])
        facet_candidates = facet_candidates[:best_facet_toadd] + facet_candidates[best_facet_toadd+1:]

    print("Completed in {} iterations".format(iterations))

    results = {
        "context_schema": context_schema,
        "facet_candidates": facet_candidates,
        "labeled_dataset": labeled_dataset,
        "cell_to_units": cell_to_units,
    }

    # for insight in iteration_insights:
    #     print(json.dumps(insight, indent=4))

    ### save the results
    taskname = task.replace(" ", "_").lower()
    parent_path = os.path.join(FRAMEWORK_PATH, f'{taskname}')
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)

    with open(os.path.join(parent_path, f"split_results_{dummy}.json"), "w") as f:
        json.dump(results, f, indent=4)

    return results

def construct_cim(task, dataset, dummy):
    important_types = IMPORTANT_TYPES_FINE
    return process_videos(task, dataset, important_types, dummy)