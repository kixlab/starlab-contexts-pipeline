from helpers.cim_scripts import build_information_units
from helpers.cim_scripts import calc_discriminativeness, calc_explained_norm, get_cell_to_units, calc_compactness
from helpers.cim_scripts import update_facet_candidates, update_facet_labels, update_labeled_dataset

from helpers.cim_scripts import FRAMEWORK_PATH

import json
import os
from collections import defaultdict

from helpers.dataset import IMPORTANT_TYPES_FINE

def get_results_path(task, version):
    taskname = task.replace(" ", "_").lower()
    parent_path = os.path.join(FRAMEWORK_PATH, f'{taskname}')
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)

    results_path = os.path.join(parent_path, f"split_results_{version}.json")
    return results_path

def load_results(task, version):
    results_path = get_results_path(task, version)
    if os.path.exists(results_path):
        with open(results_path) as f:
            return json.load(f)
    return None

def save_results(task, version, results):
    results_path = get_results_path(task, version)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

def display_sparsity(cell_to_units, display_size=1):
    cell_sizes = defaultdict(int)
    for cell, units in cell_to_units.items():
        cell_sizes[len(units)] += 1
        # if len(units) == display_size:
        #     print(cell)

    cell_sizes = sorted(cell_sizes.items(), key=lambda x: x[0])
    for size, count in cell_sizes:
        print(f"{size} unit size: {count} cells")

def compute_frontier_knapsack(labeled_dataset, piece_types, facet_candidates, max_label_count=None, over_values=True):
    """
    Compute the Pareto frontier using a 0/1 knapsack DP over discrete compactness.

    For each achievable compactness value C, compute the minimum discriminativeness D
    and record the set of facets achieving it.
    """

    # Base compactness contribution from information types (see calc_compactness)
    base_compactness = calc_compactness([], len(piece_types)) if over_values else 0

    # Weight per facet is its vocabulary size (number of labels it adds)
    facet_weights = [(calc_compactness([fc], 1) if over_values else 1) for fc in facet_candidates]
    max_weight = sum(facet_weights) if len(facet_weights) > 0 else 0

    # Evaluate discriminativeness for a given subset of indices (memoized)
    memo = {}
    def eval_subset(indices_tuple):
        if indices_tuple in memo:
            return memo[indices_tuple]
        context_schema = [facet_candidates[i] for i in indices_tuple]
        cell_to_units, relevant_units_count = get_cell_to_units(context_schema, labeled_dataset, piece_types)
        d_val = calc_discriminativeness(cell_to_units, relevant_units_count)
        explained_norm = calc_explained_norm(cell_to_units, relevant_units_count)
        memo[indices_tuple] = (d_val, explained_norm)
        return d_val, explained_norm

    # DP arrays: for each weight w, keep the best (min) D and the indices tuple
    # Initialize with +inf and no subset
    dp_D = [float("inf")] * (max_weight + 1)
    dp_explained_norm = [0] * (max_weight + 1)
    dp_subset = [None] * (max_weight + 1)

    # Base case: empty subset
    empty_tuple = tuple()
    dp_D[0], dp_explained_norm[0] = eval_subset(empty_tuple)
    dp_subset[0] = empty_tuple

    # Iterate facets (0/1 knapsack, minimizing D)
    for i, w_i in enumerate(facet_weights):
        if w_i <= 0:
            # Skip degenerate facets with no vocabulary
            continue
        if max_label_count is not None and w_i > max_label_count:
            continue
        # iterate weights backwards
        for w in range(max_weight - w_i, -1, -1):
            if dp_subset[w] is None:
                continue
            new_w = w + w_i
            new_subset = tuple(sorted(dp_subset[w] + (i,)))
            new_D, new_explained_norm = eval_subset(new_subset)
            if new_D < dp_D[new_w]:
                dp_D[new_w] = new_D
                dp_subset[new_w] = new_subset
                dp_explained_norm[new_w] = new_explained_norm

    # Build frontier results for each achievable compactness value
    frontier = []
    for w in range(0, max_weight + 1):
        subset = dp_subset[w]
        if subset is None:
            continue
        C = base_compactness + w
        D = dp_D[w]
        explained_norm = dp_explained_norm[w]
        facets = [facet_candidates[i]["id"] for i in subset]
        frontier.append({
            "compactness": C,
            "discriminativeness": D,
            "explained_norm": explained_norm,
            "facets": facets,
        })

    # Keep only the best (minimum D) per exact compactness in case of duplicates
    best_per_C = {}
    for item in frontier:
        C = item["compactness"]
        if C not in best_per_C or item["discriminativeness"] < best_per_C[C]["discriminativeness"]:
            best_per_C[C] = item

    # Return a sorted list by compactness
    result = [best_per_C[C] for C in sorted(best_per_C.keys())]

    # Smooth the frontier (do not display achievable but suboptimal compactness)
    smoothed_result = [result[0]]
    for i in range(1, len(result) - 1):
        if result[i]["discriminativeness"] - smoothed_result[-1]["discriminativeness"] < 1e-6:
            smoothed_result.append(result[i])

    return smoothed_result

def get_new_facet_candidates(task, facet_candidates, labeled_dataset, piece_types, include_cells, embedding_method, pieces_at_once, vocabulary_iterations, target_d, generation_model):
    ### Run a single iteration of facet candidate mining
    cell_to_units, relevant_units_count = get_cell_to_units(facet_candidates, labeled_dataset, piece_types)
    cur_d = calc_discriminativeness(cell_to_units, relevant_units_count)
    print(f"Discriminativeness: {cur_d}")
    display_sparsity(cell_to_units)
    if cur_d < target_d:
        return []
    new_facet_candidates = update_facet_candidates(task, facet_candidates, cell_to_units, include_cells, embedding_method, pieces_at_once, generation_model)
    new_facet_candidates = update_facet_labels(task, labeled_dataset, new_facet_candidates, vocabulary_iterations, generation_model)
    labeled_dataset = update_labeled_dataset(task, labeled_dataset, new_facet_candidates, generation_model)
    return new_facet_candidates

def process_videos_split(task, dataset, piece_types, embedding_method, extraction_model, generation_model, version):
    ### constants
    max_iterations = 5

    include_cells = 5
    pieces_at_once = 10
    vocabulary_iterations = len(dataset)
    ### Reasoning: 0.9 --> We set a higher similarity threshold to capture most of the information diversity in the dataset and remove the noise due to phrasing differences.
    information_unit_similarity_threshold=0.9
    context_length = 30 ### in seconds
    target_d = 0.8 
    ### Reasoning --> Achieving a low discriminative score is important to ensure that the context schema organizes the knowledge in a fine-grained manner.
    min_d = 0.1

    labeled_dataset = None
    facet_candidates = []

    results = load_results(task, version)
    if results is not None:
        labeled_dataset = results["labeled_dataset"]
        facet_candidates = results["facet_candidates"]

    if labeled_dataset is None:
        ### build the `information units`
        labeled_dataset = build_information_units(task, dataset, context_length, information_unit_similarity_threshold, embedding_method, extraction_model)
        save_results(task, version, {
            "context_schema": [],
            "facet_candidates": [],
            "labeled_dataset": labeled_dataset,
        })

    ### mine all facet candidates
    tries_if_empty = 0
    while tries_if_empty < max_iterations:
        new_facet_candidates = get_new_facet_candidates(task, facet_candidates, labeled_dataset, piece_types, include_cells, embedding_method, pieces_at_once, vocabulary_iterations, min_d, generation_model)
        if len(new_facet_candidates) == 0:
            tries_if_empty += 1
            continue
        tries_if_empty = 0
        facet_candidates.extend(new_facet_candidates)
        save_results(task, version, {
            "context_schema": [],
            "facet_candidates": facet_candidates,
            "labeled_dataset": labeled_dataset,
        })

    ### TODO: adjust the frontier based on IRR noise...
    frontier = compute_frontier_knapsack(labeled_dataset, piece_types, facet_candidates, max_label_count=None, over_values=True)
    context_schema = []
    found = False
    for item in frontier:
        if item["discriminativeness"] < target_d:
            for candidate in facet_candidates:
                if candidate["id"] in item["facets"]:
                    context_schema.append(candidate)
            found = True
            break
    if found:
        print(f"Found a schema with {len(context_schema)} facets that have discriminativeness score lower than {target_d}")
    else:
        print(f"No schema found with discriminativeness score lower than {target_d}")
    save_results(task, version, {
        "context_schema": context_schema,
        "facet_candidates": facet_candidates,
        "labeled_dataset": labeled_dataset,
    })

    return {
        "context_schema": context_schema,
        "facet_candidates": facet_candidates,
        "labeled_dataset": labeled_dataset,
    }

def construct_cim_split(task, dataset, embedding_method, extraction_model, generation_model, version):
    piece_types = IMPORTANT_TYPES_FINE
    results = process_videos_split(task, dataset, piece_types, embedding_method, extraction_model, generation_model, version)
    return results