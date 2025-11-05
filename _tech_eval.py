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

from collections import defaultdict

from helpers.dataset import IMPORTANT_TYPES_FINE, IMPORTANT_TYPE_DESCRIPTIONS_FINE
from helpers.dataset import get_dataset

from eval import DatasetConfig, MethodConfig, EvalConfig

from helpers.dataset import MUFFIN_TASK, CUSTOM_TASKS, CROSS_TASK_TASKS

from src.rag import generic_call_rag
from src.cim_methods import generic_call_context_similarity, generic_call_shortest_path

from eval.llm_judge import relevance_absolute_evaluation

from pydantic_models.evaluation import MetricScale

from helpers.video_scripts import get_transcript_segment

def sample_test_dataset(tasks, sample_per_task, query, n):
    test_dataset = []
    for task in tasks:
        dataset = get_dataset(task)
        sampled_tutorials = random.sample(dataset, sample_per_task)
        for tutorial in sampled_tutorials:
            info_type = random.choice(IMPORTANT_TYPES_FINE)
            info_type_description = IMPORTANT_TYPE_DESCRIPTIONS_FINE[info_type]
            info_type_description = info_type_description.strip()
            cur_query = query
            cur_query = cur_query.format(n=n, info_type=info_type, info_type_description=info_type_description)

            cur_segment = None
            if query == QUERIES[2] or query == QUERIES[3]:
                ## TODO: ensure that the `segments` attribute is present for crosstask dataset.
                selected_segment = random.choice(tutorial["segments"])
                cur_segment = {
                    "label": selected_segment["label"],
                    "content": get_transcript_segment(tutorial["transcript"], selected_segment["start"], selected_segment["end"], include_intersecting=True),
                    "start": selected_segment["start"],
                    "end": selected_segment["end"],
                }
                cur_query = cur_query.format(segment=cur_segment["label"])

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
    ### run sanity checks
    if dataset_config.query == QUERIES[2] and dataset_config.query == QUERIES[3]:
        for task in dataset_config.tasks:
            if task not in CROSS_TASK_TASKS:
                raise ValueError(f"Segment sampling is only supported for cross-task tasks. {task} is not a cross-task task.")

    test_dataset = sample_test_dataset(dataset_config.tasks, dataset_config.sample_per_task, dataset_config.query, dataset_config.n)
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
    version = method_config.version
    gen_model = method_config.gen_model
    responses = []

    ### group tests by task & restore the initial order
    tests_per_task = defaultdict(list)
    for idx, test in enumerate(test_dataset):
        task = test["task"]
        tests_per_task[task].append((idx, test))
        responses.append(None) ### placeholder for the response

    for task in tests_per_task.keys():
        dataset = get_dataset(task)
        cur_tests = [test for _, test in tests_per_task[task]]
        task_responses = method_func(task, version, dataset, cur_tests, embedding_method, gen_model, k, doc_score_threshold)
        if task_responses == None:
            continue
        initial_idxs = [initial_idx for initial_idx, _ in tests_per_task[task]]
        for idx, response in zip(initial_idxs, task_responses):
            responses[idx] = response
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
    evaluation_results = eval_func(responses_A, responses_B, test_dataset, eval_config.metric)
    
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

def main(dataset_idx, method_idx, eval_idx):
    dataset_config = DATASETS[dataset_idx]

    method_config = METHODS[method_idx]
    eval_config = EVALS[eval_idx]
    
    results = run_absolute_eval(dataset_config, method_config, eval_config)
    print(json.dumps(results, indent=4))

TECH_EVAL_PATH = "./static/results/tech_eval/"

QUERIES = [
    "Given a tutorial, retrieve top-{n} missing, but relevant `{info_type}` information for the entire tutorial.\nFollowing are examples of `{info_type}` information: \n{info_type_description}",
    "Given a tutorial, retrieve all missing, but relevant `{info_type}` information.\nFollowing are examples of `{info_type}` information: \n{info_type_description}",
    "Given a tutorial and a highlighted segment, retrieve top-{n} missing, but relevant `{info_type}` information for the highlighted segment.\nFollowing are examples of `{info_type}` information: \n{info_type_description}",
    "Given a tutorial, retrieve top-{n} missing, but relevant `{info_type}` information about the `{segment}`.\nFollowing are examples of `{info_type}` information: \n{info_type_description}",
]

DATASETS = [
    DatasetConfig(
        label="test_q1_n2",
        tasks=[MUFFIN_TASK],
        sample_per_task=2,
        query=QUERIES[0],
        n=5,
    ),
    DatasetConfig(
        label="custom_test_10_q1_n5",
        tasks=[CUSTOM_TASKS[4]],
        sample_per_task=10,
        query=QUERIES[0],
        n=5,
    ),
    DatasetConfig(
        label="custom_10_q1_n5",
        tasks=CUSTOM_TASKS,
        sample_per_task=10,
        query=QUERIES[0],
        n=5,
    ),
]

METHODS = [
    MethodConfig(
        label="RAG-openai-10-0.7",
        embedding_method="openai",
        k=8,
        doc_score_threshold=0.7,
        func=generic_call_rag,
        version=None,
        gen_model="gpt-4.1-mini-2025-04-14",
    ),
    MethodConfig(
        label="vanilla-openai",
        embedding_method="openai",
        k=None,
        doc_score_threshold=None,
        func=generic_call_rag,
        version=None,
        gen_model="gpt-4.1-mini-2025-04-14",
    ),
    MethodConfig(
        label="context-similarity-openai",
        embedding_method="openai",
        k=None,
        doc_score_threshold=None,
        func=generic_call_context_similarity,
        version="full_run_1",
        gen_model="gpt-4.1-mini-2025-04-14",
    ),
    MethodConfig(
        label="shortest-path-openai",
        embedding_method="openai",
        k=None,
        doc_score_threshold=None,
        func=generic_call_shortest_path,
        version="full_run_1",
        gen_model="gpt-4.1-mini-2025-04-14",
    ),
]

EVALS = [
    EvalConfig(
        label="relevance-absolute-evaluation-gpt-5-3",
        func=relevance_absolute_evaluation,
        metric=MetricScale.LIKERT_3,
        judge_model="gpt-5-mini-2025-08-07",
    ),
    EvalConfig(
        label="relevance-absolute-evaluation-gpt-5-binary",
        func=relevance_absolute_evaluation,
        metric=MetricScale.BINARY,
        judge_model="gpt-5-mini-2025-08-07",
    ),
    EvalConfig(
        label="faithfulness-absolute-evaluation-gpt-5",
        func=None,
        metric=None,
        judge_model="gpt-5-mini-2025-08-07",
    ),
    EvalConfig(
        label="relevance-comparative-evaluation-gpt-5",
        func=None,
        metric=MetricScale.COMPARISON,
        judge_model="gpt-5-mini-2025-08-07",
    ),
    EvalConfig(
        label="comprehensiveness-comparative-evaluation-gpt-5",
        func=None,
        metric=MetricScale.COMPARISON,
        judge_model="gpt-5-mini-2025-08-07",
    ),
]

if __name__ == "__main__":
    dataset_idx = 1
    method_idx = 2
    eval_idx = 0
    main(dataset_idx, method_idx, eval_idx)