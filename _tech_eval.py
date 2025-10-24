"""
Script that runs the technical evaluation on IR tasks

- Sample test dataset & save them
    - Test data components:
        - Task: task name
        - Tutorial: the target tutorial
        - Segment: (if needed) the target segment
        - Query: the query to be answered
- Run all the conditions on the selected inputs
- Rate/compare the results with LLM-as-a-judge (human evaluation)
- Randomly sample 10% of the results for later human evaluation
"""

import random
import os
import json

from helpers.dataset import MUFFIN_TASK, CUSTOM_TASKS, CROSS_TASK_TASKS
from helpers.dataset import IMPORTANT_TYPES
from helpers.dataset import get_dataset

from src.rag import generic_call_rag
from src.cim_methods import generic_call_context_similarity, generic_call_shortest_path

TECH_EVAL_PATH = "./static/results/tech_eval/"

DATASETS = [
    {
        "label": "test_q0_n5",
        "tasks": [MUFFIN_TASK],
        "query_idx": 0,
        "N_idx": 0,
    },
    {
        "label": "cross-task_q0_n5",
        "tasks": CROSS_TASK_TASKS,
        "query_idx": 0,
        "N_idx": 0,
    },
    {
        "label": "custom_q0_n5",
        "tasks": CUSTOM_TASKS[:4],
        "query_idx": 0,
        "N_idx": 0,
    },
]

METHODS = {
    "rag": {
        "label": "RAG-bert-10-0.7",
        "embedding_method": "bert",
        "k": 10,
        "doc_score_threshold": 0.7,
        "func": generic_call_rag,
    },
    "vanilla": {
        "label": "vanilla-bert",
        "embedding_method": "bert",
        "k": None,
        "doc_score_threshold": None,
        "func": generic_call_rag,
    },
    "context_similarity": {
        "label": "context-similarity-bert",
        "embedding_method": "bert",
        "k": None,
        "doc_score_threshold": None,
        "func": generic_call_context_similarity,
    },
    "shortest_path": {
        "label": "shortest-path-bert",
        "embedding_method": "bert",
        "k": None,
        "doc_score_threshold": None,
        "func": generic_call_shortest_path,
    },
}

EVALS = {
    "relevance_abs": {
        "label": "relevance-analysis-abs",
        "func": None,
    },
    "relevance_rel": {
        "label": "relevance-analysis-rel",
        "func": None,
    },
    "faithfulness": {
        "label": "faithfulness-analysis",
        "func": None,
    },
    "comprehensiveness": {
        "label": "comprehensiveness-analysis",
        "func": None,
    },
}

QUERIES = [
    "Given a tutorial with the highlighted segment, retrieve top-{N} missing, but relevant {TYPE} information for the segment.",
    "Given a tutorial, retrieve top-{N} missing, but relevant {TYPE} information for the entire tutorial.",
    "Given a tutorial, retrieve all missing, but relevant {TYPE} information.",
]

Ns = [5, 20]

TYPES = IMPORTANT_TYPES

def sample_test_dataset(tasks, sample_per_task, query_idx, N_idx):
    test_dataset = []
    N = Ns[N_idx]
    query = QUERIES[query_idx]
    for task in tasks:
        dataset = get_dataset(task)
        sampled_tutorials = random.sample(dataset, sample_per_task)
        for tutorial in sampled_tutorials:
            TYPE = random.choice(TYPES)
            
            cur_query = query
            if query_idx < 2:
                cur_query = cur_query.format(N=N, TYPE=TYPE)
            else:
                cur_query = cur_query.format(TYPE=TYPE)

            cur_segment = None
            if query_idx < 1:
                cur_segment = "TODO"
            
            test_dataset.append({
                "task": task,
                "tutorial": tutorial,
                "segment": cur_segment,
                "query": cur_query,
                "n": N,
            })

    return test_dataset

def construct_test_dataset(dataset_config):
    label = dataset_config["label"]
    test_dataset_path = os.path.join(TECH_EVAL_PATH, label + ".json")
    if os.path.exists(test_dataset_path):
        with open(test_dataset_path, "r") as f:
            return json.load(f)

    test_dataset = sample_test_dataset(dataset_config["tasks"], dataset_config["query_idx"], dataset_config["N_idx"])
    with open(test_dataset_path, "w") as f:
        json.dump(test_dataset, f, indent=4)
    return test_dataset

def test_dataset_statistics(test_dataset):
    stats_per_query = {}
    for test in test_dataset:
        query = test["query"]
        if query not in stats_per_query:
            stats_per_query[query] = {
                "count": 0,
                "tasks": [],
                "segments": [],
            }
        stats_per_query[query]["count"] += 1
        stats_per_query[query]["tasks"].append(test["task"])
        stats_per_query[query]["segments"].append(test["segment"])
    print("Size: ", len(test_dataset))
    for query, stats in stats_per_query.items():
        print("-"*20)
        print(query)
        print(f"count: {stats['count']}")
        print(f"tasks: {stats['tasks']}")
        print(f"segments: {stats['segments']}")
        print("-"*20)

def run_method(method_config, test_dataset):
    method_func = method_config["func"]
    embedding_method = method_config["embedding_method"]
    k = method_config["k"]
    doc_score_threshold = method_config["doc_score_threshold"]
    responses = []
    for test in test_dataset:
        task = test["task"]
        tutorial = test["tutorial"]
        segment = test["segment"]
        query = test["query"]
        n = test["n"]
        dataset = get_dataset(task)
        response = method_func(embedding_method, task, dataset, tutorial, query, segment, n, k, doc_score_threshold)
        responses.append(response)
    return responses

def run_eval_abs(dataset_config, method_config, eval_config):
    dataset_label = dataset_config["label"]
    eval_label = eval_config["label"]
    method_label = method_config["label"]

    results_path = os.path.join(TECH_EVAL_PATH, f"{dataset_label}_{method_label}_{eval_label}.json")

    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            return json.load(f)

    print("Constructing test dataset...")
    test_dataset = construct_test_dataset(dataset_config)
    test_dataset_statistics(test_dataset)

    print("Running method...")
    responses = run_method(method_config, test_dataset)
    

    print("Running eval...")
    eval_func = eval_config["func"]
    results = eval_func(responses, test_dataset)
    
    results = {
        "responses": responses,
        "dataset_config": dataset_config,
        "method_config": method_config,
        "eval_config": eval_config,
    }
    
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    return results

def run_eval_rel(dataset_config, method_configs, eval_config):
    ### Get responses from all the methods and run the comparative eval_config on them.
    pass

def main():
    dataset_config = DATASETS[0] ### test_q0_n5
    ## dataset_config = DATASETS[1] ### cross-task_q0_n5
    ## dataset_config = DATASETS[2] ### custom_q0_n5

    method_config = METHODS["rag"]
    eval_config = EVALS["relevance_abs"]
    
    results = run_eval_abs(dataset_config, method_config, eval_config)
    print(json.dumps(results, indent=4))

if __name__ == "__main__":
    main()