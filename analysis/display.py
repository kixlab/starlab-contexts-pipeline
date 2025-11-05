import json
from collections import defaultdict
import numpy as np

from helpers import count_tokens
from helpers.cim_scripts import get_cell_to_units, calc_discriminativeness, calc_explained_norm

def show_task_stats(task, result, relevant_types):
    dataset = result["labeled_dataset"]
    units = defaultdict(int)
    relevant_units = defaultdict(int)
    tokens_per_tutorial = []
    relevant_tokens_per_tutorial = []
    count_units = 0
    count_pieces = 0
    count_relevant_units = 0
    count_relevant_pieces = 0
    count_tutorials = len(dataset)
    for tutorial in dataset:
        tokens = 0
        relevant_tokens = 0
        for piece in tutorial["pieces"]:
            cur_tokens = count_tokens(piece["content"])
            if piece["content_type"] in relevant_types:
                relevant_tokens += cur_tokens
                count_relevant_pieces += 1
                relevant_units[piece["unit_id"]] += 1
            count_pieces += 1
            units[piece["unit_id"]] += 1
            tokens += cur_tokens
        tokens_per_tutorial.append(tokens)
        relevant_tokens_per_tutorial.append(relevant_tokens)
    
    count_units = len(units)
    count_relevant_units = len(relevant_units)

    print(f"Task: {task}")
    print(f"Total units: {count_units}")
    print(f"Total pieces: {count_pieces}")
    print(f"Total relevant units: {count_relevant_units}")
    print(f"Total relevant pieces: {count_relevant_pieces}")
    print(f"Total tutorials: {count_tutorials}")
    unit_relevance_ratio = -1
    if count_units > 0:
        unit_relevance_ratio = round(count_relevant_units / (count_units), 2)
    print(f"Unit relevance ratio: {unit_relevance_ratio}")

    piece_relevance_ratio = -1
    if count_pieces > 0:
        piece_relevance_ratio = round(count_relevant_pieces / (count_pieces), 2)
    print(f"Piece relevance ratio: {piece_relevance_ratio}")
    
    print(f"Total & Average & Std tokens per tutorial: {np.sum(tokens_per_tutorial)}, {np.average(tokens_per_tutorial)}, {np.std(tokens_per_tutorial)}")
    print(f"Total & Average & Std relevant tokens per tutorial: {np.sum(relevant_tokens_per_tutorial)}, {np.average(relevant_tokens_per_tutorial)}, {np.std(relevant_tokens_per_tutorial)}")

    facets = result["facet_candidates"]
    count_facets = len(facets)
    count_labels = sum([len(facet["vocabulary"]) for facet in facets])
    avg_labels_per_facet = -1
    if count_facets > 0:
        avg_labels_per_facet = round(count_labels / count_facets, 2)
    print(f"Total facets: {count_facets}")
    print(f"Total labels: {count_labels}")
    print(f"Average labels per facet: {avg_labels_per_facet}")

    print("--------------------------------")
    cell_to_units, relevant_units_count = get_cell_to_units(facets, dataset, relevant_types)
    cur_d = calc_discriminativeness(cell_to_units, relevant_units_count)
    explained_norm = calc_explained_norm(cell_to_units, relevant_units_count)
    print("Discriminativeness: ", cur_d)
    print("Explained norm: ", explained_norm)
    print()


def display_tutorial_contexts(tutorial, include_keys=None, include_content_types=None):
    """
    print in markdown table format where columns are keys and rows are values
    """
    columns = defaultdict(list)
    
    columns['info_type'] = []
    columns['content'] = []

    for piece in tutorial['pieces']:
        if include_content_types is not None and piece['content_type'] not in include_content_types:
            continue
        columns['info_type'].append(piece['content_type'])
        if 'content' in columns:
            columns['content'].append(piece['content'])
        for key, value in piece['labels'].items():
            if include_keys is None or key in include_keys:
                columns[key].append(value[-1])
    contents = columns['content']
    del columns['content']
    columns['content'] = contents
    
    markdown_table = "| " + " | ".join(columns.keys()) + " |\n"
    markdown_table += "| " + " | ".join(["---"] * len(columns)) + " |\n"
    for i in range(len(columns[list(columns.keys())[0]])):
        markdown_table += "| " + " | ".join([columns[key][i] for key in columns.keys()]) + " |\n"
    markdown_table += "| " + " | ".join(["---"] * len(columns)) + " |\n"
    print(markdown_table)

def display_units(dataset, include_keys=None, include_content_types=None):
    import matplotlib.pyplot as plt
    units = defaultdict(list)
    for tutorial in dataset:
        for piece in tutorial['pieces']:
            if include_content_types is not None and piece['content_type'] not in include_content_types:
                continue
            units[piece['unit_id']].append(piece)
    
    ### sort in descending order of number of pieces
    units = sorted(units.items(), key=lambda x: len(x[1]), reverse=True)

    ### show distribution of pieces across units (#of pieces vs #of units)
    lengths = defaultdict(int)
    for _, pieces in units:
        lengths[len(pieces)] += 1
    
    print(json.dumps(lengths, indent=4))

    # for unit_id, pieces in units:
    #     print(f"Unit {unit_id} ({len(pieces)} pieces) ")
    #     for piece in pieces:
    #         print(f"  - {piece['content_type']}: {piece['content']}")
    #     print()

def display_type_distribution(dataset, piece_types):
    per_type = defaultdict(int)

    for tutorial in dataset:
        for piece in tutorial['pieces']:
            if piece['content_type'] in piece_types:
                per_type[piece['content_type']] += 1
    
    ### sort in descending order of number of pieces
    per_type = sorted(per_type.items(), key=lambda x: x[1], reverse=True)

    print(json.dumps(per_type, indent=4))