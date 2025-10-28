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
from helpers.dataset import get_dataset
from src.framework import construct_cim
from helpers.cim_scripts import get_facet_name

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
    for facet in schema:
        key = get_facet_name(facet)
        if key in embeddings:
            raise ValueError(f"Facet {key} already exists")
        embeddings[key] = {"": blank}
        for v in facet["vocabulary"]:
            if v["label"] in embeddings[key]:
                raise ValueError(f"Label {v['label']} already exists")
            embeddings[key][v["label"]] = perform_embedding(
                embedding_method, [f"{v['label']}: {v['definition']}"]
            )[0]
    return embeddings

def score_context_similarity(a_context, b_context, facet_value_embeddings, hard_matching=False):
    score = 0.0
    for key in facet_value_embeddings:
        value_a = a_context.get(key)
        value_b = b_context.get(key)
        if value_a is None or value_b is None:
            continue
        if hard_matching:
            score += 1.0 if value_a == value_b else 0.0
        else:
            score += float(np.dot(facet_value_embeddings[key][value_a], facet_value_embeddings[key][value_b]))
    return score / float(len(facet_value_embeddings))

def context_similarity_retrieval(embedding_method, task, dataset, tutorial, segment, info_type, n):
    ### segment can be None and n can be None
    construction_results = construct_cim(task, dataset, "final")

    facet_value_embeddings = build_facet_value_embeddings(embedding_method, construction_results["context_schema"])

    facet_keys = set(facet_value_embeddings)

    cim = normalize_context_values(construction_results["labeled_dataset"], facet_keys)

    labeled_tutorial = next(t for t in cim if t["url"] == tutorial["url"])

    covered_context_signatures = {
        format_context_signature(p["labels"], facet_keys)
        for p in labeled_tutorial["pieces"]
        if p["content_type"] == info_type
    }
    for covered_context_signature in covered_context_signatures:
        print(covered_context_signature)
    print("--------------------------------")
    targets_iter = (
        p["labels"] for p in labeled_tutorial["pieces"]
        if p["content_type"] != info_type
    )

    context_bucket_map = defaultdict(lambda: {"count": 0, "pieces": [], "labels": None})
    total_missing_contexts = 0
    for target_context in targets_iter:
        context_signature = format_context_signature(target_context, facet_keys)
        if context_signature in covered_context_signatures:
            continue
        bucket = context_bucket_map[context_signature]
        bucket["count"] += 1
        bucket["labels"] = target_context
        total_missing_contexts += 1
    
    print(json.dumps(context_bucket_map, indent=4))

    for other_idx, other in enumerate(cim):
        if other["url"] == tutorial["url"]:
            continue
        for piece in other["pieces"]:
            if piece["content_type"] == info_type:
                continue
            context_signature = format_context_signature(piece["labels"], facet_keys)
            if context_signature in context_bucket_map:
                target_context = context_bucket_map[context_signature]["labels"]
                context_bucket_map[context_signature]["pieces"].append({
                    "score": score_context_similarity(target_context, piece["labels"], facet_value_embeddings),
                    "source_doc_idx": other_idx,
                    "content": piece["content"],
                })
                break
    
    context_candidates = []
    
    for _, info in context_bucket_map.items():
        if not info["pieces"]:
            continue
        top_unit = max(info["pieces"], key=lambda p: p["score"])
        context_candidates.append({
            "score": top_unit["score"],
            "significance": info["count"] / total_missing_contexts,
            "content": top_unit["content"],
            "source_doc_idx": top_unit["source_doc_idx"],
        })
        

    context_candidates = sorted(context_candidates, key=lambda x: x["score"], reverse=True)

    return [
        {"content": c["content"], "source_doc_idx": c["source_doc_idx"]}
        for c in context_candidates[:n]
    ] ### list of responses: {"content": "...", "source_doc_idx": "..."}

generic_call_context_similarity = lambda embedding_method, task, dataset, tutorial, segment, query, info_type, n, k, doc_score_threshold: context_similarity_retrieval(embedding_method, task, dataset, tutorial, segment, info_type, n)

def shortest_path_retrieval(embedding_method, task, dataset, tutorial, segment, info_type, n):
    ### segment can be None and n can be None
    dataset = get_dataset(task)
    cim = construct_cim(task, "final")

    ### single response: {"content": "...", "source_doc_idx": "..."}
    pass

generic_call_shortest_path = lambda embedding_method, task, dataset, tutorial, segment, query, info_type, n, k, doc_score_threshold: shortest_path_retrieval(embedding_method, task, dataset, tutorial, segment, info_type, n)
