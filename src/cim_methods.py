"""
Implementation of two methods based on the CIM:
- Context similarity retrieval: retrieves IUs with most similar contexts to the target tutorial/segment contexts
    - find all the contexts that appear in the tutorial but has missing info.
    - for all pieces of the type of info_type, compute the context similarity.
- Shortest path retrieval: retrieves IUs that have shortest path to the target tutorial/segment contexts
"""
import json
import numpy as np
from collections import defaultdict

from helpers import perform_embedding
from src.framework_split import construct_cim_split, construct_cim_split_conservative
from src.framework_iter import construct_cim_iter

from queue import Queue

def normalize_context_values(cim, facet_keys):
    for tutorial in cim:
        ### TODO: assign the last label for now, but adjust later
        for piece in tutorial["pieces"]:
            for cur_key, cur_labels in piece["labels"].items():
                piece["labels"][cur_key] = cur_labels[-1]
        
        # ### fill in empty labels
        # print(f"Tutorial: {tutorial['url']}")
        # for cur_idx, piece in enumerate(tutorial["pieces"]):
        #     print(f"\t{piece['content_type']}: {piece['content']}")
        #     for cur_key, cur_label in piece["labels"].items():
        #         if cur_key not in facet_keys:
        #             continue
        #         print(f"\t\t{cur_key}: {cur_label}")
        #         if cur_label == "":
        #             ### find the closest label from the left or right
        #             min_distance = 1e9
        #             for l_idx in range(cur_idx - 1, -1, -1):
        #                 if tutorial["pieces"][l_idx]["labels"][cur_key] != "":
        #                     cur_label = tutorial["pieces"][l_idx]["labels"][cur_key]
        #                     min_distance = cur_idx - l_idx
        #                     break
        #             for r_idx in range(cur_idx + 1, len(tutorial["pieces"])):
        #                 if tutorial["pieces"][r_idx]["labels"][cur_key] != "":
        #                     if r_idx - cur_idx <= min_distance:
        #                         cur_label = tutorial["pieces"][r_idx]["labels"][cur_key]
        #                         min_distance = r_idx - cur_idx
        #                     break
        #             if cur_label == "":
        #                 print(f"Label {cur_key} is empty for piece {cur_idx} {tutorial['url']}")
        #             tutorial["pieces"][cur_idx]["labels"][cur_key] = cur_label
        #         print(f"\t\t--> {cur_key}: {cur_label}")
    return cim

def format_context_signature(context, facet_keys):
    return " ".join(
        f"{k}-{v}"
        for k, v in context.items()
        if k in facet_keys
    )

def build_facet_value_embeddings(embedding_method, schema):
    embeddings = {}
    blank = perform_embedding(embedding_method, [" "])[0]
    ### TODO: blank may not be the best "empty" representative. For example, when calculating similarity between contexts, if "empty" is aligned with "non-empty", it may need to incur a big penalty, but it may end up too small depending on what the other value is. 
    for facet in schema:
        key = facet["id"]
        if key in embeddings:
            raise ValueError(f"Facet {key} already exists")
        embeddings[key] = {
            "": blank,
            "na": blank,
        }
        for v in facet["vocabulary"]:
            # if v["label"] in embeddings[key] and v["label"] != "na":
            #     raise ValueError(f"WARNING: Label {v['label']} already exists {key}")
            if v["label"] in embeddings[key]:
                continue
            embeddings[key][v["label"]] = perform_embedding(
                embedding_method, [f"{v['label']}: {v['definition']}"]
            )[0]
    return embeddings

def score_context_similarity(a_context, b_context, facet_value_embeddings, hard_matching=False):
    score = 0.0
    common_keys = 0
    for key in facet_value_embeddings:
        value_a = a_context.get(key)
        value_b = b_context.get(key)
        if value_a is None or value_b is None:
            continue
        common_keys += 1
        if hard_matching:
            score += 1.0 if value_a == value_b else 0.0
        else:
            # print(key, value_a, value_b)
            cur_score = float(np.dot(facet_value_embeddings[key][value_a], facet_value_embeddings[key][value_b]))
            score += cur_score
            # print(f"Score: {float(cur_score):.4f}")
    if common_keys == 0:
        return 0.0
    return score / float(common_keys)

def get_missing_contexts(labeled_tutorial, segment, info_type, facet_keys):
    covered_context_signatures = set([
        format_context_signature(p["labels"], facet_keys)
        for p in labeled_tutorial["pieces"]
        if p["content_type"] == info_type
    ])

    target_pieces = (
        p for p in labeled_tutorial["pieces"]
        if p["content_type"] != info_type
    )

    if segment is not None:
        target_pieces = (
            p for p in labeled_tutorial["pieces"]
            if p["start"] >= segment["start"] and p["end"] <= segment["end"]
        )

    context_bucket_map = defaultdict(lambda: {"count": 0, "length": 0.0, "pieces": [], "augmenting_pieces": []})

    for target_piece in target_pieces:
        target_context = target_piece["labels"]
        context_signature = format_context_signature(target_context, facet_keys)
        if context_signature in covered_context_signatures:
            continue
        bucket = context_bucket_map[context_signature]
        bucket["length"] += target_piece["end"] - target_piece["start"]
        bucket["pieces"].append(target_piece)
        bucket["count"] += 1
    
    return context_bucket_map

def select_candidates(candidates, score_threshold):
    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
    response = []

    for c in candidates:
        if score_threshold is not None and c["score"] < score_threshold:
            break
        response.append({
            "score": c["score"],
            "significance": c["significance"],
            "content": c["content"], 
            "raw_context": c["raw_context"],
            "content_type": c["content_type"],
            "source_doc_idx": c["source_doc_idx"],
            "source_piece_idx": c["source_piece_idx"],
        })
    if len(response) == 0:
        print("WARNING: No candidates found with score >= ", score_threshold)
        return []
    else:
        print (f"Found {len(response)} candidates with score >= {score_threshold}", f"The worst score is {candidates[len(response)-1]['score']:.4f}")

    response = sorted(response, key=lambda x: x["score"], reverse=True)
    for c in response:
        del c["significance"]
        del c["score"]
    return response


def context_similarity_retrieval(cim, facet_value_embeddings, tutorial, segment, info_type, score_threshold):
    facet_keys = set(facet_value_embeddings.keys())
    labeled_tutorial = next(t for t in cim if t["url"] == tutorial["url"])

    context_bucket_map = get_missing_contexts(labeled_tutorial, segment, info_type, facet_keys)
    total_augmentable_length = sum(bucket["length"] for bucket in context_bucket_map.values())

    for other in cim:
        if other["url"] == tutorial["url"]:
            continue
        for piece in other["pieces"]:
            if piece["content_type"] != info_type:
                continue
            final_score = 0.0
            closest_context_signature = None
            for context_signature in context_bucket_map.keys():
                for target_piece in context_bucket_map[context_signature]["pieces"]:
                    score = score_context_similarity(target_piece["labels"], piece["labels"], facet_value_embeddings)
                    if score > final_score:
                        final_score = score
                        closest_context_signature = context_signature
            if closest_context_signature is None:
                print(f"WARNING: No closest context signature found for piece {piece['content_type']} {piece['content']}")
                continue
            context_bucket_map[closest_context_signature]["augmenting_pieces"].append({
                "score": final_score,
                "source_doc_idx": other['url'],
                "source_piece_idx": piece["piece_id"],
                "content": piece["content"],
                "raw_context": piece["raw_context"],
                "content_type": piece["content_type"],
            })
    
    candidates = []
    
    still_missing_contexts = 0
    for _, info in context_bucket_map.items():
        if len(info["augmenting_pieces"]) == 0:
            still_missing_contexts += 1
            continue
        top_unit = max(info["augmenting_pieces"], key=lambda p: p["score"])
        candidates.append({
            "score": top_unit["score"],
            "significance": info["length"] / total_augmentable_length,
            "source_doc_idx": top_unit["source_doc_idx"],
            "source_piece_idx": top_unit["source_piece_idx"],
            "content": top_unit["content"],
            "raw_context": top_unit["raw_context"],
            "content_type": top_unit["content_type"],
        })
        
    return select_candidates(candidates, score_threshold)


def shortest_path_retrieval(cim, facet_value_embeddings, tutorial, segment, info_type, score_threshold):
    ### segment can be None and n can be None
    facet_keys = set(facet_value_embeddings.keys())
    labeled_tutorial = next(t for t in cim if t["url"] == tutorial["url"])

    context_bucket_map = get_missing_contexts(labeled_tutorial, segment, info_type, facet_keys)

    graph = defaultdict(lambda: defaultdict(int))
    for t in cim:
        if t["url"] == tutorial["url"]:
            continue
        for p in t["pieces"]:
            if p["content_type"] != info_type:
                continue
            context_signature = format_context_signature(p["labels"], facet_keys)
            if context_signature in context_bucket_map:
                graph[context_signature][p["unit_id"]] += 1
                graph[p["unit_id"]][context_signature] += 1
    
    min_distances = defaultdict(list)
    for target_context_signature in context_bucket_map.keys():
        ### bfs
        min_distances[target_context_signature].append((0, 1e9)) ### (distance, min_weight on the path)
        queue = Queue()
        queue.put((target_context_signature, 0, 1e9))
        visited = set([target_context_signature])
        while queue:
            v, distance, min_weight = queue.get()
            if min_weight < min_distances[v][-1][1]:
                continue
            for u, weight in graph[v].items():
                cur_min_weight = min(min_weight, weight)
                if u not in visited:
                    visited.add(u)
                    queue.put((u, distance + 1, cur_min_weight))
                    min_distances[u].append((distance + 1, cur_min_weight))
                elif min_distances[u][-1][0] == distance + 1 and min_distances[u][-1][1] < cur_min_weight:
                    min_distances[u][-1] = (distance + 1, cur_min_weight)
                    queue.put((u, distance + 1, cur_min_weight))
    
    candidates = []
    for t in cim:
        if t["url"] == tutorial["url"]:
            continue
        for p in t["pieces"]:
            if p["unit_id"] not in min_distances:
                continue

            min_distance, max_min_weight = 1e9, 0
            for distance, min_weight in min_distances[p["unit_id"]]:
                if distance < min_distance:
                    min_distance = distance
                    max_min_weight = min_weight
                elif distance == min_distance and min_weight > max_min_weight:
                    max_min_weight = min_weight
            candidates.append({
                "score": min_distance,
                "content": p["content"],
                "raw_context": p["raw_context"],
                "source_doc_idx": t["url"],
                "significance": max_min_weight,
            })
    return select_candidates(candidates, score_threshold)

def run_cim_method(task, dataset, tests, func, config):
    embedding_method = config["embedding_method"]
    version = config["version"]
    score_threshold = config["response_score_threshold"]

    construction_results = None
    if version is None:
        ### use the latest version (e.g., `full_run_X`)
        for idx in range(0, 100):
            cur_version = f"full_run_{idx}"
            cur = construct_cim_split_conservative(task, cur_version)
            if cur is not None:
                construction_results = cur
                version = cur_version
    else:
        construction_results = construct_cim_split_conservative(task, version)

    print(f"Using version {version} for task {task}")
    
    if construction_results is None:
        print(f"Construction results are not available for task {task}")
        return []

    ### use all `facet_candidates` instead of `context_schema`
    facet_value_embeddings = build_facet_value_embeddings(embedding_method, construction_results["facet_candidates"])
    
    facet_keys = set(facet_value_embeddings.keys())

    cim = normalize_context_values(construction_results["labeled_dataset"], facet_keys)

    responses = []

    for test in tests:
        tutorial = test["tutorial"]
        segment = test["segment"]
        info_type = test["info_type"]

        response = func(cim, facet_value_embeddings, tutorial, segment, info_type, score_threshold)
        responses.append(response)
    return responses

generic_call_context_similarity = lambda task, dataset, tests, config: run_cim_method(task, dataset, tests, context_similarity_retrieval, config)

generic_call_shortest_path = lambda task, dataset, tests, config: run_cim_method(task, dataset, tests, shortest_path_retrieval, config)