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

from helpers.dataset import IMPORTANT_TYPES, IMPORTANT_TYPE_DESCRIPTIONS
from helpers.dataset import get_dataset

from eval import DatasetConfig, MethodConfig, EvalConfig

from helpers.dataset import MUFFIN_TASK, CUSTOM_TASKS, CROSS_TASK_TASKS

from src.rag import generic_call_rag
from src.cim_methods import generic_call_context_similarity, generic_call_shortest_path

from eval.llm_judge import relevance_absolute_evaluation

from pydantic_models.evaluation import MetricScale

def sample_test_dataset(tasks, sample_per_task, sample_segment_per_tutorial, query, n):
    test_dataset = []
    for task in tasks:
        dataset = get_dataset(task)
        sampled_tutorials = random.sample(dataset, sample_per_task)
        for tutorial in sampled_tutorials:
            info_type = random.choice(IMPORTANT_TYPES)
            info_type_description = IMPORTANT_TYPE_DESCRIPTIONS[info_type]
            info_type_description = info_type_description.strip()
            cur_query = query
            cur_query = cur_query.format(n=n, info_type=info_type)
            cur_query = cur_query + f"\nFollowing are examples of {info_type} information: \n{info_type_description}" ### add definition for the kind of information to retrieve

            cur_segment = None
            if sample_segment_per_tutorial > 0:
                cur_segment = f"TODO: sample {sample_segment_per_tutorial} segments per tutorial" ### let's do left;right timestamps?
            
            test_dataset.append({
                "task": task,
                "tutorial": tutorial,
                "segment": cur_segment,
                "query": cur_query,
                "info_type": info_type,
                "n": n,
            })

    return test_dataset

def construct_test_dataset(dataset_config):
    label = dataset_config.label
    test_dataset_path = os.path.join(TECH_EVAL_PATH, label + ".json")
    if os.path.exists(test_dataset_path):
        with open(test_dataset_path, "r") as f:
            return json.load(f)

    test_dataset = sample_test_dataset(dataset_config.tasks, dataset_config.sample_per_task, dataset_config.sample_segment_per_tutorial, dataset_config.query, dataset_config.n)
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
    method_func = method_config.func
    embedding_method = method_config.embedding_method
    k = method_config.k
    doc_score_threshold = method_config.doc_score_threshold
    responses = []
    for test in test_dataset:
        info_type = test["info_type"]
        task = test["task"]
        tutorial = test["tutorial"]
        segment = test["segment"]
        query = test["query"]
        n = test["n"]
        dataset = get_dataset(task)
        response = method_func(embedding_method, task, dataset, tutorial, segment, query, info_type, n, k, doc_score_threshold)
        responses.append(response)
    return responses

def run_absolute_eval(dataset_config, method_config, eval_config):
    dataset_label = dataset_config.label
    eval_label = eval_config.label
    method_label = method_config.label

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
    eval_func = eval_config.func
    evaluation_results = eval_func(responses, test_dataset, eval_config.metric)
    
    results = {
        "evaluation_type": "absolute",
        "evaluation_results": evaluation_results,
        "responses": responses,
        "dataset_config": dataset_config.to_json(),
        "method_config": method_config.to_json(),
        "eval_config": eval_config.to_json(),
    }
    
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    return results

def run_comparative_eval(dataset_config, method_config_A, method_config_B, eval_config):
    dataset_label = dataset_config["label"]
    eval_label = eval_config["label"]
    method_labels = [method_config_A["label"], method_config_B["label"]]
    ### sort method_labels alphabetically
    method_labels.sort()
    method_labels = '_'.join(method_labels)
    results_path = os.path.join(TECH_EVAL_PATH, f"{dataset_label}_{method_labels}_{eval_label}.json")
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            return json.load(f)
    
    print("Constructing test dataset...")
    test_dataset = construct_test_dataset(dataset_config)
    test_dataset_statistics(test_dataset)

    print("Running methods...")
    responses_A = run_method(method_config_A, test_dataset)
    responses_B = run_method(method_config_B, test_dataset)

    print("Running eval...")
    eval_func = eval_config["func"]
    evaluation_results = eval_func(responses_A, responses_B, test_dataset)
    
    results = {
        "evaluation_type": "comparative",
        "evaluation_results": evaluation_results,
        "responses_A": responses_A,
        "responses_B": responses_B,
        "dataset_config": dataset_config,
        "method_config_A": method_config_A,
        "method_config_B": method_config_B,
        "eval_config": eval_config,
    }
    
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    return results

def main():
    dataset_config = DATASETS[0] ### test_q0_n5
    ## dataset_config = DATASETS[1] ### cross-task_q0_n5
    ## dataset_config = DATASETS[2] ### custom_q0_n5

    method_config = METHODS[2]
    eval_config = EVALS[0]
    
    results = run_absolute_eval(dataset_config, method_config, eval_config)
    print(json.dumps(results, indent=4))

TECH_EVAL_PATH = "./static/results/tech_eval/"

QUERIES = [
    "Given a tutorial with the highlighted segment, retrieve top-{n} missing, but relevant {info_type} information for the segment.",
    "Given a tutorial, retrieve top-{n} missing, but relevant {info_type} information for the entire tutorial.",
    "Given a tutorial, retrieve all missing, but relevant {info_type} information.",
]

DATASETS = [
    DatasetConfig(
        label="test_q1_n2",
        tasks=[MUFFIN_TASK],
        sample_per_task=2,
        query=QUERIES[1],
        sample_segment_per_tutorial=0, ### specify if query_idx = 0
        n=5,
    ),
    DatasetConfig(
        label="cross-task_q0_n5",
        tasks=CROSS_TASK_TASKS,
        sample_per_task=5,
        query=QUERIES[0],
        sample_segment_per_tutorial=1,
        n=5,
    ),
    DatasetConfig(
        label="custom_q0_n5",
        tasks=CUSTOM_TASKS[:4], ### TODO: actually sample 4 tasks
        sample_per_task=5,
        query=QUERIES[0],
        sample_segment_per_tutorial=1,
        n=5,
    ),
]

METHODS = [
    MethodConfig(
        label="RAG-bert-10-0.7",
        embedding_method="bert",
        k=10,
        doc_score_threshold=0.7,
        func=generic_call_rag,
    ),
    MethodConfig(
        label="vanilla-bert",
        embedding_method="bert",
        k=None,
        doc_score_threshold=None,
        func=generic_call_rag,
    ),
    MethodConfig(
        label="context-similarity-bert",
        embedding_method="bert",
        k=None,
        doc_score_threshold=None,
        func=generic_call_context_similarity,
    ),
    MethodConfig(
        label="shortest-path-bert",
        embedding_method="bert",
        k=None,
        doc_score_threshold=None,
        func=generic_call_shortest_path,
    ),
]

EVALS = [
    EvalConfig(
        label="relevance-absolute-evaluation-3",
        func=relevance_absolute_evaluation,
        metric=MetricScale.LIKERT_3,
    ),
    EvalConfig(
        label="relevance-absolute-evaluation-binary",
        func=relevance_absolute_evaluation,
        metric=MetricScale.BINARY,
    ),
    EvalConfig(
        label="faithfulness-absolute-evaluation",
        func=None,
        metric=None,
    ),
    EvalConfig(
        label="relevance-comparative-evaluation",
        func=None,
        metric=MetricScale.COMPARISON,
    ),
    EvalConfig(
        label="comprehensiveness-comparative-evaluation",
        func=None,
        metric=MetricScale.COMPARISON,
    ),
]



if __name__ == "__main__":
    main()