from helpers.cim_scripts import build_information_units_v0
from helpers.cim_scripts import relative_improvement, calc_discriminativeness, calc_compactness, macro_pruning, get_cell_to_units
from helpers.cim_scripts import update_facet_candidates, update_facet_labels, update_labeled_dataset

from helpers.cim_scripts import FRAMEWORK_PATH

import json
import os

from helpers.dataset import IMPORTANT_TYPES_FINE

def process_videos_iter(task, dataset, piece_types, version):
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
    embedding_method = "openai"
    pieces_at_once = 5
    vocabulary_iterations = len(dataset)

    taskname = task.replace(" ", "_").lower()
    parent_path = os.path.join(FRAMEWORK_PATH, f'{taskname}')
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)

    results_path = os.path.join(parent_path, f"iter_results_{version}.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            return json.load(f)

    ### build the `information units`
    labeled_dataset = build_information_units_v0(task, dataset, context_length, information_unit_similarity_threshold, embedding_method)

    ### Greedy Algorithm for constructing the schema:

    facet_candidates = []
    context_schema = []

    iteration_insights = []
    iterations = 0
    
    base_cell_to_units, base_relevant_units_count = get_cell_to_units(context_schema, labeled_dataset, piece_types)
    base_d = calc_discriminativeness(base_cell_to_units, base_relevant_units_count)
    base_c = calc_compactness(context_schema, len(piece_types))

    while True:

        iterations += 1
        if iterations > max_iterations:
            print("WARNING: Maximum number of iterations reached")
            break

        ### run pruning every `pruning_interval` iterations
        if iterations % pruning_interval == 0 and not skip_pruning:
            original_length = len(context_schema)
            while len(context_schema) > 1 and len(context_schema) > original_length - max_macro_pruning_len:
                new_context_schema, removed_facet = macro_pruning(
                    context_schema, labeled_dataset, piece_types, pruning_threshold
                )
                if len(new_context_schema) < len(context_schema):
                    context_schema = new_context_schema
                    if removed_facet is not None:
                        facet_candidates.append(removed_facet)
                    continue
                else:
                    break   
            iteration_insights.append({
                "type": "macro_pruning",
                "iteration": iterations,
                "original_length": original_length,
                "new_length": len(context_schema),
                "removed_facet": (None if removed_facet is None else removed_facet["title"]),
            })

        cell_to_units, relevant_units_count = get_cell_to_units(context_schema, labeled_dataset, piece_types)

        ### get new facet candidates
        ### TODO: update the entire facet candidates list, but tag which ones are new & perform vocabulary building and labeling for the new ones only
        new_facet_candidates = update_facet_candidates(task, cell_to_units, include_cells, embedding_method, pieces_at_once)
        new_facet_candidates = update_facet_labels(task, labeled_dataset, new_facet_candidates, vocabulary_iterations)
        labeled_dataset = update_labeled_dataset(task, labeled_dataset, new_facet_candidates)

        facet_candidates.extend(new_facet_candidates)
        
        if len(facet_candidates) == 0:
            print("WARNING: No facet candidates left")
            break    

        prev_d = calc_discriminativeness(cell_to_units, relevant_units_count)
        prev_c = calc_compactness(context_schema, len(piece_types))
        o_biggest = -float("inf")
        best_c = 1
        best_d = prev_d
        best_facet_toadd = None
        for i, facet_candidate in enumerate(facet_candidates):
            ### get the labeling by the context schema
            candidate_context_schema = context_schema + [facet_candidate]
            candidate_cell_to_units, candidate_relevant_units_count = get_cell_to_units(candidate_context_schema, labeled_dataset, piece_types)
            ### calculate the discriminative and compactness
            d = calc_discriminativeness(candidate_cell_to_units, candidate_relevant_units_count)
            c = calc_compactness(candidate_context_schema, len(piece_types))
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
    }

    # for insight in iteration_insights:
    #     print(json.dumps(insight, indent=4))
    
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    return results

def construct_cim_iter(task, dataset,version):
    piece_types = IMPORTANT_TYPES_FINE
    return process_videos_iter(task, dataset, piece_types, version)