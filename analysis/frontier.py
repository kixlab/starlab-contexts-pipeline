import json
import matplotlib.pyplot as plt
import numpy as np
import os
import math
from collections import defaultdict


from src.framework_split import construct_cim_split_conservative
from src.framework_split import compute_frontier_knapsack
from helpers.nlp import clustering_custom

def find_piece_by_unit(dataset, unit_id, info_types):
    for tutorial in dataset:
        for piece in tutorial['pieces']:
            if piece['content_type'] not in info_types:
                continue
            if piece['unit_id'] == unit_id:
                return piece['content']
    return None

def get_facets(facet_candidates, facet_titles):
    facets = []
    for facet_title in facet_titles:
        for facet in facet_candidates:
            if facet["title"] == facet_title:
                facets.append(facet)
    return facets

def distribution_of_labels(results, top_k=5):
    labels_count = defaultdict(int)
    facet_name_count = defaultdict(int)
    facet_presence_count = defaultdict(set)
    for task, result in results.items():
        facet_candidates = result["facet_candidates"]
        for facet in facet_candidates:
            facet_name_count[facet["title"]] += 1
            labels_count[len(facet["vocabulary"])] += 1
            facet_presence_count[facet["title"]].add(task)
    
    for facet_name in facet_name_count.keys():
        ### average across tasks
        facet_name_count[facet_name] /= len(facet_presence_count[facet_name])
    
    top_k_facet_names = sorted(facet_name_count.items(), key=lambda x: x[1], reverse=True)[:top_k]
    

    top_k_facet_names = [x[0] for x in top_k_facet_names]
    plt.bar(top_k_facet_names, [facet_name_count[x] for x in top_k_facet_names])
    plt.show()

    plt.bar(labels_count.keys(), labels_count.values())
    plt.show()

def calc_sparsity(cell_to_units, display_size=-1):
    cell_sizes = defaultdict(int)
    for cell, units in cell_to_units.items():
        cell_sizes[len(units)] += 1
        if len(units) == display_size:
            print(cell)
            for unit in units:
                print(unit)

    cell_sizes = sorted(cell_sizes.items(), key=lambda x: x[0])
    
    for size, count in cell_sizes:
        print(f"{size} unit size: {count} cells")

def show_frontier_item(item, facet_candidates):
    agg_facets = []
    # agg_vocab = []
    for facet_id in item["facets"]:
        for facet in facet_candidates:
            if facet["id"] == facet_id:
                agg_facets.append(facet["title"])
                # for label in facet["vocabulary"]:
                #     agg_vocab.append(label["label"])
    print("ELBOW FACETS: ", agg_facets)
    # print("ELBOW VOCAB: ", agg_vocab)

def get_colors(n):
    colors = plt.cm.rainbow(np.linspace(0, 1, n))
    ### make the color 50% transparent
    colors = [(color[0], color[1], color[2], 0.5) for color in colors]
    return colors

def plot_frontier(plt, x, y, label, color, linestyle, axvlines=[]):
    # for _x, _y, _label in axvlines:
    #     plt.axvline(x=_x, color=color, linestyle=':')
        ## plt.text(_x, _y, _label)
    plt.plot(x, y, label=label, color=color, linestyle=linestyle)

def interpolate_frontier(x, y, y_new):
    """
    Interpolate the frontier. Assumes x is non-decreasing.
    """
    for i in range(len(x)):
        if abs(y[i] - y_new) < 1e-6:
            return x[i]
    
    for i in range(len(x) - 1):
        y_l = y[i]
        y_h = y[i + 1]
        if y_l > y_h:
            y_l, y_h = y_h, y_l
        if y_l - y_new < 1e-6 and y_h - y_new > 1e-6:
            return x[i] + (x[i + 1] - x[i]) * (y_new - y[i]) / (y[i + 1] - y[i])
    return x[-1] + 1.0

def plot_frontiers(
    frontiers, y_axis, cutoff_x, output_path
):
    plt.figure(figsize=(10, 10))
    y_titles = {
        "discriminativeness": "Discriminativeness",
        "explained_norm": "Explained Variance",
    }
    best_cs = []

    colors = get_colors(len(frontiers))

    for info, color in zip(frontiers, colors):
        frontier = info["frontier"]
        elbows = info["elbows"]
        label = info["task"]

        x = [item["compactness"] for item in frontier]
        y = [item[y_axis] for item in frontier]
        if cutoff_x is not None:
            x = [x for x in x if x <= cutoff_x]
            y = y[:len(x)]
        
        best_c = -1
        prev_dist = float("inf")
        
        for item in frontier:
            cur_dist = float("inf")
            if y_axis == "explained_norm":
                cur_dist = abs(item["explained_norm"] - 0.9)
            if y_axis == "discriminativeness":
                cur_dist = abs(item["discriminativeness"] - 1)
            if cur_dist < prev_dist:
                best_c = item["compactness"]
                prev_dist = cur_dist
        
        best_cs.append(best_c)
        axvlines = []
        if prev_dist < 1e-1:
            axvlines.append((best_c, 1, f"d~{1:.2f}"))
        for elbow, w_d in elbows:
            axvlines.append((elbow["compactness"], elbow[y_axis], f"{w_d:.2f}"))

        linestyle = "-"
        if label.startswith("common_"):
            linestyle = "--"
        plot_frontier(plt, x, y, label, color, linestyle, axvlines=axvlines)

    avg_best_c = np.average(best_cs)
    std_best_c = np.std(best_cs)
    print(f"Average optimal compactness: {avg_best_c:.2f} (std: {std_best_c:.2f})")

    plt.legend()
    plt.xlabel("Schema Complexity")
    plt.ylabel(y_titles[y_axis])
    plt.title("Pareto Frontier")
    plt.xlim(right=(cutoff_x if cutoff_x is not None else 100))
    plt.ylim(bottom=0)
    plt.savefig(output_path)
    plt.close()

def find_elbow(frontier, w_d):
    if len(frontier) == 0:
        return None
    w_c = 1 - w_d
    optimal_idx = -1
    best_score = float("inf")
    for idx, item in enumerate(frontier):
        cur_d = item["discriminativeness"]
        cur_c = item["compactness"]
        score = w_d * cur_d + w_c * cur_c
        if score < best_score:
            best_score = score
            optimal_idx = idx
    return frontier[optimal_idx]

def find_elbows(frontier, base_d, inc=0.01):
    cur_elbows = []
    w_d = base_d
    while w_d < 1.0:
        elbow_item = find_elbow(frontier, w_d)
        if elbow_item is None:
            continue
        cur_elbows.append((elbow_item, w_d))
        w_d += inc
    return cur_elbows

def get_frontiers(results, cur_types, w_d, max_label_count=None, over_values=True):
    frontiers = []
    print("Found", len(results), "tasks")
    for task, result in results.items():
        dataset = result["labeled_dataset"]
        facet_candidates = result["facet_candidates"]
        frontier = compute_frontier_knapsack(dataset, cur_types, facet_candidates, max_label_count, over_values)

        if w_d is not None:
            elbow_item = find_elbow(frontier, w_d)
            frontiers.append({
                "task": task,
                "frontier": frontier,
                "elbows": [(elbow_item, w_d)],
            })
        else:
            frontiers.append({
                "task": task,
                "frontier": frontier,
                "elbows": [],
            })
    return frontiers

def plot_frontiers_facets(results, piece_types, elbow_d, y_axis, output_folder):
    max_label_count = None
    over_values = False
    cutoff_x = 50
    
    frontiers = get_frontiers(results, piece_types, elbow_d, max_label_count, over_values)
    
    output_path = os.path.join(output_folder, "frontier_facets.png")
    plot_frontiers(frontiers, y_axis, cutoff_x, output_path)

def plot_frontiers_labels(results, piece_types, elbow_d, y_axis, output_folder):
    max_label_count = None
    over_values = True
    cutoff_x = None
    
    frontiers = get_frontiers(results, piece_types, elbow_d, max_label_count, over_values)

    output_path = os.path.join(output_folder, "frontier_labels.png")
    plot_frontiers(frontiers, y_axis, cutoff_x, output_path)

def get_available_results(tasks, dummies):
    results = {}
    for task, version in zip(tasks, dummies):
        result = construct_cim_split_conservative(task, version)
        if result is None:
            continue
        task_name = task.lower().replace(" ", "_") + "_" + version
        results[task_name] = result
    return results

def classify_facet_candidates(results, similarity_threshold, embedding_method):
    """
    Classify the facet candidates into common vs unique to the task.
    """
    updated_results = {}
    facet_id_to_class = {}
    facet_id_to_task = {}
    facet_texts_per_id = {}
    all_unique_tasks = set()
    for task, result in results.items():
        all_unique_tasks.add(task)
        facet_candidates = result["facet_candidates"]

        for facet in facet_candidates:
            ## facet_text = f"{facet['title']}: {facet['definition']}"
            facet_text = facet['title']
            facet_texts_per_id[facet['id']] = facet_text
            facet_id_to_task[facet['id']] = task

    ## cluster facet_texts into clusters
    ## assign a cluster a class `common` or `unique` depending on the size of the cluster (if cluster size = 1, it is unique, otherwise it is common)
    clusters = clustering_custom(facet_texts_per_id.values(), similarity_threshold, embedding_method)
    cluster_sizes = defaultdict(list)
    for facet_id, cluster in zip(facet_texts_per_id.keys(), clusters):
        cluster_sizes[cluster].append(facet_id)
    
    unique_tasks_cluster_count = defaultdict(int)
    for cluster, facet_ids in cluster_sizes.items():
        unique_tasks = set()
        for facet_id in facet_ids:
            unique_tasks.add(facet_id_to_task[facet_id])
        unique_tasks_cluster_count[len(unique_tasks)] += len(facet_ids)
        # cur_class = "common"
        # if len(unique_tasks) == 1:
        #     cur_class = "unique"
        cur_class = "unique"
        if len(unique_tasks) >= len(all_unique_tasks) // 2 + 1:
            cur_class = "common"
        
        for facet_id in facet_ids:
            facet_id_to_class[facet_id] = cur_class

    print(f"Total facet candidates: {len(facet_texts_per_id)}")
    print(json.dumps(unique_tasks_cluster_count, indent=4))

    for task, result in results.items():
        task_common = f"common_{task}"
        task_unique = f"unique_{task}"
        facet_candidates = result["facet_candidates"]

        candidates_common = []
        candidates_unique = []
        for facet in facet_candidates:
            if facet_id_to_class[facet['id']] == "common":
                ### add the common to unique as well
                candidates_common.append(facet)
                candidates_unique.append(facet)
                print("common", task, facet['title'], facet['definition'])
            else:
                candidates_unique.append(facet)
                print("unique", task, facet['title'], facet['definition'])

        updated_results[task_common] = {
            **result,
            "facet_candidates": candidates_common,
        }
        updated_results[task_unique] = {
            **result,
            "facet_candidates": candidates_unique,
        }
    return updated_results
