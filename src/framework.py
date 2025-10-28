import json
import os

from helpers.dataset import MUFFIN_TASK, CUSTOM_TASKS, CROSS_TASK_TASKS
from helpers.dataset import IMPORTANT_TYPES

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

    cell_to_units = {}
    relevant_units = set()
    for video in dataset:
        for piece in video["pieces"]:
            if piece["content_type"] not in important_piece_types:
                continue
            cell_id = f"<{piece['content_type']}>"
            for facet in context_schema:
                cell_id += f"<{get_label(piece, get_facet_name(facet))}>"
            
            if cell_id not in cell_to_units:
                cell_to_units[cell_id] = {}
            
            if piece["unit_id"] not in cell_to_units[cell_id]:
                cell_to_units[cell_id][piece["unit_id"]] = []
            cell_to_units[cell_id][piece["unit_id"]].append(piece)
            relevant_units.add(piece["unit_id"])

    ### TODO: remove sparsity check later
    # sparsity = len(important_piece_types)
    # for facet in context_schema:
    #     sparsity *= len(facet["vocabulary"]) + 1
    # sparsity = len(cell_to_units) / sparsity * 100
    # print("Sparsity %: ", sparsity)
    
    return cell_to_units, len(relevant_units)


from helpers.bert import bert_embedding, clustering_custom, find_most_distant_pair
from prompts.framework import form_information_units

def build_information_units_v0(task, dataset, context_length, information_unit_similarity_threshold, dummy=""):
    
    taskname = task.replace(" ", "_").lower()
    parent_path = os.path.join(FRAMEWORK_PATH, f'{taskname}')
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)

    path = os.path.join(parent_path, f"information_units_{dummy}.json")
    if os.path.exists(path):
        with open(path) as f:
            dataset = json.load(f)
            return dataset
    
    for video_idx, video in enumerate(dataset):
        if "pieces" in video:
            pieces = video["pieces"]
        else:
            ### forming the information units (conceptually should be easily redefinable)
            pieces = form_information_units(task, video['transcript'])
            video['pieces'] = []
            for i, piece in enumerate(pieces):
                video['pieces'].append({
                    "piece_id": f"piece_{video_idx}_{i}",
                    **piece,
                    "labels": {},
                    "context_before": "",
                    "context_after": "",
                })

    ### add context before and after to each piece
    for video_idx, video in enumerate(dataset):
        for i, piece in enumerate(video['pieces']):
            context_before = ""
            context_after = ""
            if i - context_length < 0:
                context_before = "[Tutorial start]"
            l = max(0, i-context_length)
            r = min(len(video['pieces']), i+context_length+1)
            if i + context_length >= len(video['pieces']):
                context_after = "[Tutorial end]"
            for piece_ in video['pieces'][l:i]:
                context_before += piece_["content"] + " "
            for piece_ in video['pieces'][i+1:r]:
                context_after += piece_["content"] + " "
            piece["context_before"] = context_before
            piece["context_after"] = context_after


    ### TODO: maybe cluster for each type of information separately?
    all_pieces = []
    for video_idx, video in enumerate(dataset):
        for i, piece in enumerate(video['pieces']):
            piece["piece_id"] = f"piece_{video_idx}_{i}"
            all_pieces.append(piece)

    #### cluster similar pieces in `all_pieces`
    information_units = {}

    unit_labels = clustering_custom([piece["content"] for piece in all_pieces], information_unit_similarity_threshold)
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

    with open(path, "w") as f:
        json.dump(dataset, f, indent=4)

    return dataset

from prompts.framework import form_codebook, label_transcript_pieces

def build_codebook_v0(task, dataset, facet):
    """
    similar to VideoMix --> iteratively build the codebook
    TODO: may need to restrict the number of videos considered
    """
    # facet_name = get_facet_name(facet)
    # taskname = task.replace(" ", "_").lower()

    # parent_path = os.path.join(FRAMEWORK_PATH, f'{taskname}')
    # if not os.path.exists(parent_path):
    #     os.makedirs(parent_path)

    # path = os.path.join(parent_path, f"{facet_name}_v0.json")
    # if os.path.exists(path):
    #     with open(path) as f:
    #         schema = json.load(f)
    #         return schema

    ### iteratively build the context schema
    for video in dataset:
        vocabulary = form_codebook(task, video["transcript"], facet)
        facet["vocabulary"] = vocabulary

    # with open(path, "w") as f:
    #     json.dump(schema, f, indent=4)

    return facet

def label_based_on_codebook_v0(task, dataset, facet):
    """
    label the dataset based on the codebook
    TODO: label multiple times per video
    """
    
    facet_name = get_facet_name(facet)
    # taskname = task.replace(" ", "_").lower()

    # parent_path = os.path.join(FRAMEWORK_PATH, f'{taskname}')
    # if not os.path.exists(parent_path):
    #     os.makedirs(parent_path)
    
    # path = os.path.join(parent_path, f"labeling_{facet_name}_v0.json")
    # if os.path.exists(path):
    #     with open(path) as f:
    #         dataset = json.load(f)
    #         return dataset

    for video in dataset:
        if "pieces" in video:
            if len(video["pieces"]) == 0:
                continue
            if facet_name in video["pieces"][0]["labels"]:
                ### skip if already labeled across `schema`
                continue
        else:
            print(f"WARNING: No pieces found for video {video['title']}")
            continue

        labeled_pieces = label_transcript_pieces(task, video["pieces"], facet)

        if len(labeled_pieces) != len(video["pieces"]):
            print(f"STRONG WARNING: {len(labeled_pieces)} != {len(video['pieces'])}")
        for piece_idx, piece in enumerate(video["pieces"]):
            if facet_name not in piece["labels"]:
                piece["labels"][facet_name] = []
            piece["labels"][facet_name].append(labeled_pieces[piece_idx])
    
    # if not os.path.exists(path):
    #     with open(path, "w") as f:
    #         json.dump(dataset, f, indent=4)

    return dataset

from prompts.framework import form_segmentation_facet_candidates, combine_segmentation_facet_candidates

def build_facet_candidates_v0(task, cell_to_units, prev_facet_candidates, include_cells):

    cell_ids = list(cell_to_units.keys())

    cell_ids.sort(key=lambda x: len(cell_to_units[x]), reverse=True)
    
    cell_ids = cell_ids[:include_cells]

    all_candidates = []
    all_candidates.extend(prev_facet_candidates)
    for cell_id in cell_ids:
        if len(cell_to_units[cell_id]) <= 1:
            break
        combined_pieces = []
        for unit_id in cell_to_units[cell_id]:
            combined_pieces.extend(cell_to_units[cell_id][unit_id])

        piece_texts = [(piece["context_before"] + " " + piece["content"] + " " + piece["context_after"]).lower() for piece in combined_pieces]
        p1, p2 = find_most_distant_pair(piece_texts)
        
        new_facet_candidates = form_segmentation_facet_candidates(task, [combined_pieces[p1], combined_pieces[p2]])
        all_candidates.extend(new_facet_candidates)

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
    
    # print("Entropy check: ", total_entropy, check_entropy, relevant_units_count, n_c_list)

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
        return context_schema
    return context_schema[:facet_to_remove] + context_schema[facet_to_remove+1:]

def update_facet_candidates(
    task, cell_to_units, facet_candidates, include_cells, dummy=""
):
    """
    Update the facet candidates and the labeled dataset.
    """

    if include_cells == 0:
        return facet_candidates

    taskname = task.replace(" ", "_").lower()
    parent_path = os.path.join(FRAMEWORK_PATH, f'{taskname}')
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)
    
    path = os.path.join(parent_path, f"facet_candidates_{dummy}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)

    facet_candidates = build_facet_candidates_v0(task, cell_to_units, facet_candidates, include_cells)

    with open(path, "w") as f:
        json.dump(facet_candidates, f, indent=4)

    return facet_candidates

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


def process_videos_approach_1(task, dataset, important_piece_types, dummy):
    ### constants
    max_iterations = 100
    pruning_interval = 5
    pruning_threshold = 1
    max_macro_pruning_len = 2
    skip_pruning = True
    stopping_delta_threshold = 0.1
    include_cells = 10
    information_unit_similarity_threshold=0.8
    context_length = 5

    app1_dummy = "app1_" + dummy

    ### build the `information units`
    labeled_dataset = build_information_units_v0(task, dataset, context_length, information_unit_similarity_threshold, app1_dummy)

    ### Greedy Algorithm for constructing the schema:

    facet_candidates = []
    context_schema = []

    iteration_insights = []
    iterations = 0
    
    base_cell_to_units, base_relevant_units_count = get_cell_to_units(context_schema, labeled_dataset, important_piece_types)
    base_d = calc_discriminativeness(base_cell_to_units, base_relevant_units_count)
    base_c = calc_compactness(context_schema, len(important_piece_types))

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

    approach_1_results = {
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

    with open(os.path.join(parent_path, f"approach_1_results_{dummy}.json"), "w") as f:
        json.dump(approach_1_results, f, indent=4)

    return approach_1_results

### OUTPUT
import os

def construct_cim(task, dataset,dummy):
    important_types = IMPORTANT_TYPES
    return process_videos_approach_1(task, dataset, important_types, dummy)

import tiktoken
import numpy as np    

def count_tokens(tasks):
    for task in tasks:
        tokens_per_tutorial = []
        dataset = get_dataset(task)
        for tutorial in dataset:
            content = tutorial["content"]
            encoding = tiktoken.encoding_for_model("gpt-4")
            n_tokens = len(encoding.encode(content))
            tokens_per_tutorial.append(n_tokens)
    
        print(task)
        print(np.average(tokens_per_tutorial))
        print(np.sum(tokens_per_tutorial))