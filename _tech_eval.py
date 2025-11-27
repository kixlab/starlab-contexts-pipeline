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
import itertools
import numpy as np
from collections import defaultdict
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
from scipy.stats import shapiro
from scipy.stats import wilcoxon

from helpers.dataset import IMPORTANT_TYPES_FINE, IMPORTANT_TYPE_DESCRIPTIONS_FINE

from eval import DATASETS, METHODS, EVALS, TECH_EVAL_PATH, RESPONSE_ITEMS_CAP

from eval.llm_judge import (get_scores_average, 
                            get_scores_info_type,
                            get_absolute_scores_precision,
                            get_absolute_scores_ap,
                            get_absolute_scores_ndcg,
                            get_absolute_scores_precision_length_adjusted,
                            get_absolute_scores_rr,
                            )

from helpers.video_scripts import get_transcript_segment

from helpers.cim_scripts import extract_pieces

def sample_test_dataset(tasks, sample_per_task, query):
    test_dataset = []
    for task in tasks:
        dataset = extract_pieces(task)
        sampled_tutorials = random.sample(dataset, sample_per_task)
        for tutorial in sampled_tutorials:
            info_type = random.choice(IMPORTANT_TYPES_FINE)
            cur_query = query.format(info_type=info_type, info_type_description=IMPORTANT_TYPE_DESCRIPTIONS_FINE[info_type].strip())

            cur_segment = None
            if "segment" in cur_query:
                if "segments" not in tutorial:
                    raise ValueError(f"Segment sampling is only supported for CROSS_TASK_TASKS. {task} is not a cross-task task.")
                selected_segment = random.choice(tutorial["segments"])
                cur_segment = {
                    "label": selected_segment["label"],
                    "content": get_transcript_segment(tutorial["pieces"], selected_segment["start"], selected_segment["end"], include_intersecting=True),
                    "start": selected_segment["start"],
                    "end": selected_segment["end"],
                }

            test_dataset.append({
                "task": task,
                "tutorial": tutorial,
                "segment": cur_segment,
                "query": cur_query,
                "info_type": info_type,
            })

    return test_dataset

def construct_test_dataset(dataset_config):
    label = dataset_config.label
    test_dataset_path = os.path.join(TECH_EVAL_PATH, label + ".json")
    if os.path.exists(test_dataset_path):
        with open(test_dataset_path, "r") as f:
            return json.load(f)

    tasks = dataset_config.tasks
    sample_per_task = dataset_config.sample_per_task
    query = dataset_config.query

    test_dataset = sample_test_dataset(tasks, sample_per_task, query)
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
                "info_types": [],
            }
        stats_per_query[query]["count"] += 1
        stats_per_query[query]["tasks"].append(test["task"])
        stats_per_query[query]["segments"].append(test["segment"])
        stats_per_query[query]["info_types"].append(test["info_type"])
    print("Size: ", len(test_dataset))
    for query, stats in stats_per_query.items():
        print("-"*20)
        print(query)
        print(f"count: {stats['count']}")
        print(f"tasks: {stats['tasks']}")
        print(f"info_types: {stats['info_types']}")
        print(f"segments: {stats['segments']}")
        print("-"*20)

def run_method(method_config, dataset_config):

    method_label = method_config.label
    dataset_label = dataset_config.label
    results_path = os.path.join(TECH_EVAL_PATH, f"{dataset_label}_{method_label}.json")
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            responses = json.load(f)
        return responses

    method_func = method_config.func
    embedding_method = method_config.embedding_method
    doc_k = method_config.doc_k
    doc_score_threshold = method_config.doc_score_threshold
    version = method_config.version
    response_score_threshold = method_config.response_score_threshold
    gen_model = method_config.gen_model
    responses = []

    test_dataset = construct_test_dataset(dataset_config)

    ### group tests by task & restore the initial order
    tests_per_task = defaultdict(list)
    for idx, test in enumerate(test_dataset):
        task = test["task"]
        tests_per_task[task].append((idx, test))
        responses.append(None) ### placeholder for the response

    for task in tests_per_task.keys():
        dataset = extract_pieces(task)
        cur_tests = [test for _, test in tests_per_task[task]]
        task_responses = method_func(task, dataset, cur_tests, {
            "embedding_method": embedding_method,
            "doc_k": doc_k,
            "doc_score_threshold": doc_score_threshold,
            "response_score_threshold": response_score_threshold,
            "gen_model": gen_model,
            "version": version,
        })
        if task_responses == None:
            continue
        initial_idxs = [initial_idx for initial_idx, _ in tests_per_task[task]]
        for idx, response in zip(initial_idxs, task_responses):
            responses[idx] = response

    with open(results_path, "w") as f:
        json.dump(responses, f, indent=4)
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
    eval_responses = run_method(method_config, dataset_config)

    if eval_responses is None or len(eval_responses) != len(test_dataset):
        raise ValueError(f"Could not properly run method {method_config.label} for {dataset_config.label}")

    print("Response lengths: ", [len(eval_response_i) for eval_response_i in eval_responses])
    for i in range(len(eval_responses)):
        eval_responses[i] = eval_responses[i][:RESPONSE_ITEMS_CAP]

    print("Running eval...")
    eval_func = eval_config.func
    eval_metric = eval_config.metric
    eval_judge_model = eval_config.judge_model
    eval_n = eval_config.n
    results = eval_func(eval_responses, test_dataset, {
        "metric": eval_metric,
        "judge_model": eval_judge_model,
        "n": eval_n,
    })
    
    results = {
        "evaluation_type": "absolute",
        "results": results,
        "eval_responses": eval_responses,
        "test_dataset": test_dataset,
        "dataset_config": dataset_config.to_dict(),
        "method_config": method_config.to_dict(),
        "eval_config": eval_config.to_dict(),
    }
    
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    return results

def run_comparative_eval(dataset_config, method_config_A, method_config_B, eval_config):
    dataset_label = dataset_config.label
    eval_label = eval_config.label
    method_labels = [method_config_A.label, method_config_B.label]
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

    print("Running method A...")
    eval_responses_A = run_method(method_config_A, dataset_config)
    
    if eval_responses_A is None or len(eval_responses_A) != len(test_dataset):
        raise ValueError(f"Could not properly run method {method_config_A.label} for {dataset_config.label}")

    print("Response lengths A: ", [len(eval_response_i) for eval_response_i in eval_responses_A])
    for i in range(len(eval_responses_A)):
        eval_responses_A[i] = eval_responses_A[i][:RESPONSE_ITEMS_CAP]

    print("Running method B...")
    eval_responses_B = run_method(method_config_B, dataset_config)

    if eval_responses_B is None or len(eval_responses_B) != len(test_dataset):
        raise ValueError(f"Could not properly run method {method_config_B.label} for {dataset_config.label}")
    
    print("Response lengths B: ", [len(eval_response_i) for eval_response_i in eval_responses_B])
    for i in range(len(eval_responses_B)):
        eval_responses_B[i] = eval_responses_B[i][:RESPONSE_ITEMS_CAP]

    print("Running eval...")
    eval_func = eval_config.func
    eval_metric = eval_config.metric
    eval_judge_model = eval_config.judge_model
    eval_n = eval_config.n
    results = eval_func(eval_responses_A, eval_responses_B, test_dataset, {
        "metric": eval_metric,
        "judge_model": eval_judge_model,
        "n": eval_n,
    })
    
    results = {
        "evaluation_type": "comparative",
        "results": results,
        "eval_responses": eval_responses_A,
        "eval_responses_other": eval_responses_B,
        "test_dataset": test_dataset,
        "dataset_config": dataset_config.to_dict(),
        "method_config": method_config_A.to_dict(),
        "method_config_other": method_config_B.to_dict(),
        "eval_config": eval_config.to_dict(),
    }
    
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    return results

def t_test(scores_a, scores_b, path):
    
    np_a = np.array(scores_a)
    np_b = np.array(scores_b)

    ### draw the distribution of score_a-score_b
    diff = np_a - np_b
    plt.hist(diff)
    plt.savefig(path)
    plt.close()
    ### check if diff is normally distributed
    shapiro_test = shapiro(diff)
    if shapiro_test.pvalue > 0.05:
        ### Paired t-test
        print("\t\tDiff is normally distributed", shapiro_test)
        t_stat, p_value = ttest_rel(np_a, np_b, nan_policy="omit", alternative="two-sided")
        return t_stat, p_value, "paired t-test"
    else:
        ### Wilcoxon Signed-Rank Test
        print("\t\tDiff is not normally distributed", shapiro_test)
        t_stat, p_value = wilcoxon(np_a, np_b, correction=True, alternative="two-sided")
        return t_stat, p_value, "wilcoxon"

def output_representative_examples(combined_results, sampled_indexes):
    ## print representative results per score
    results_across_methods = defaultdict(dict)
    for method_key in combined_results.keys():
        for eval_key in combined_results[method_key].keys():
            cur_combined_results = combined_results[method_key][eval_key]
            for idx in sampled_indexes:
                if idx >= len(cur_combined_results["test_dataset"]):
                    continue
                cur_test = cur_combined_results["test_dataset"][idx]
                if cur_test["info_type"] != "Method - Instruction":
                    continue
                results_across_methods[f"{idx+1}_{eval_key}"]["task"] = cur_test["task"]
                results_across_methods[f"{idx+1}_{eval_key}"]["tutorial"] = cur_test["tutorial"]["url"]
                results_across_methods[f"{idx+1}_{eval_key}"]["segment"] = cur_test["segment"]["label"] if cur_test["segment"] is not None else "-"
                results_across_methods[f"{idx+1}_{eval_key}"]["query"] = cur_test["query"]
                
                cur_result = cur_combined_results["results"][idx]
                if cur_result is None or len(cur_result) == 0:
                    cur_result = "-"
                else:
                    cur_result = "\n\n".join([f"{item['rating']}\n({item['reasoning']})" for item in cur_result])
                cur_response = cur_combined_results["eval_responses"][idx]
                if cur_response is None or len(cur_response) == 0:
                    cur_response = "-"
                else:
                    cur_response = "\n\n".join([item["content"] for item in cur_response])
                results_across_methods[f"{idx+1}_{eval_key}"][method_key] = {
                    "result": cur_result,
                    "response": cur_response,
                }
    

    ### print as .md table
    column_keys = ["Index", "Task", "Tutorial", "Segment", "Query"] + [f"`{method_key}`" for method_key in combined_results.keys()]
    print("<md>")
    print("| " + " | ".join(column_keys) + " |")
    print("| " + " | ".join(["---"] * len(column_keys)) + " |")
    for index, results in results_across_methods.items():
        if len(results) == 0:
            continue
        evals_values = {
            "Index": f"`{index}`",
            "Task": f"`{results['task']}`".replace("\n", "<br>"),
            "Tutorial": f"`{results['tutorial']}`".replace("\n", "<br>"),
            "Segment": results['segment'].replace("\n", "<br>"),
            "Query": results['query'].replace("\n", "<br>"),
        }
        responses_values = {
            "Index": "",
            "Task": "",
            "Tutorial": "",
            "Segment": "",
            "Query": "",
        }
        for method_key in combined_results.keys():
            if method_key in results:
                evals_values[method_key] = f"{results[method_key]['result']}".replace("\n", "<br>")
                responses_values[method_key] = f"{results[method_key]['response']}".replace("\n", "<br>")
            else:
                evals_values[method_key] = "-"
                responses_values[method_key] = "-"
        print("| " + " | ".join([str(evals_values[key]) for key in evals_values.keys()]) + " |")
        print("| " + " | ".join([str(responses_values[key]) for key in responses_values.keys()]) + " |")
    print("</md>")

def output_results(combined_results, score_types):
    for method in combined_results.keys():
        for evaluation in combined_results[method].keys():
            cur_combined_results = combined_results[method][evaluation]
            print(f"Method: {method}, Evaluation: {evaluation}")
            for score_type in score_types.keys():
                func = score_types[score_type]
                cur_scores = func(
                    cur_combined_results["results"],
                    cur_combined_results["eval_responses"],
                    cur_combined_results["test_dataset"],
                )
                print(f"\t{np.mean(cur_scores):.2f} ({np.std(cur_scores):.2f}) ({score_type})")

                cur_scores_per_info_type = defaultdict(list)
                for idx in range(len(cur_combined_results["test_dataset"])):
                    cur_info_type = cur_combined_results["test_dataset"][idx]["info_type"]
                    cur_scores_per_info_type[cur_info_type].append(cur_scores[idx])

                for info_type, scores in cur_scores_per_info_type.items():
                    print(f"\t\t{np.mean(scores):.2f} ({np.std(scores):.2f}) ({info_type})")

    print()
    print("-"*100)

    folder_path = os.path.join(TECH_EVAL_PATH, "diff_distribution")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for method_1 in combined_results.keys():
        for method_2 in combined_results.keys():
            if method_1 == method_2:
                continue
            common_results = combined_results[method_1].keys() & combined_results[method_2].keys()
            if len(common_results) == 0:
                continue
            
            print(f"Method: {method_1}, Method: {method_2}")
            for score_type in score_types.keys():
                func = score_types[score_type]
                scores_a = []
                scores_b = []
                scores_a_per_info_type = defaultdict(list)
                scores_b_per_info_type = defaultdict(list)
                for common_result_key in common_results:
                    cur_combined_results_1 = combined_results[method_1][common_result_key]
                    cur_combined_results_2 = combined_results[method_2][common_result_key]
                    
                    cur_scores_1 = func(
                        cur_combined_results_1["results"],
                        cur_combined_results_1["eval_responses"],
                        cur_combined_results_1["test_dataset"],
                    )
                    scores_a.extend(cur_scores_1)

                    cur_scores_2 = func(
                        cur_combined_results_2["results"],
                        cur_combined_results_2["eval_responses"],
                        cur_combined_results_2["test_dataset"],
                    )
                    scores_b.extend(cur_scores_2)
                    for idx in range(len(cur_combined_results_2["test_dataset"])):
                        cur_info_type_1 = cur_combined_results_1["test_dataset"][idx]["info_type"]
                        cur_info_type_2 = cur_combined_results_2["test_dataset"][idx]["info_type"]
                        scores_a_per_info_type[cur_info_type_1].append(cur_scores_1[idx])
                        scores_b_per_info_type[cur_info_type_2].append(cur_scores_2[idx])
                print(f"\tScore Type: {score_type}")
                file_name = f"{score_type}_{method_1}_{method_2}.png"            
                path = os.path.join(folder_path, file_name)
                stat, p_value, test_type = t_test(scores_a, scores_b, path)
                print(f"\t\tStat: {stat:.4f}, P-value: {p_value:.4f}, Test Type: {test_type}")
                print(f"\t\tMean: {np.mean(scores_a):.2f}, Std: {np.std(scores_a):.2f}")
                print(f"\t\tMean: {np.mean(scores_b):.2f}, Std: {np.std(scores_b):.2f}")
                # for info_type in IMPORTANT_TYPES_FINE:
                #     print(f"\t\tInfo Type: {info_type}")
                #     if info_type not in scores_a_per_info_type:
                #         print("\t\t\tNo data")
                #         continue
                #     print(f"\t\t\tMean: {np.mean(scores_a_per_info_type[info_type]):.4f}, Std: {np.std(scores_a_per_info_type[info_type]):.4f}")
                #     print("\t\t\tScores: ", [f"{score:.4f}" for score in scores_a_per_info_type[info_type]])
                #     print(f"\t\t\tMean: {np.mean(scores_b_per_info_type[info_type]):.4f}, Std: {np.std(scores_b_per_info_type[info_type]):.4f}")
                #     print("\t\t\tScores: ", [f"{score:.4f}" for score in scores_b_per_info_type[info_type]])
                #     print("-"*20)
            print("-"*20)
    
    output_representative_examples(combined_results, range(200))

def main():

    n = 5

    score_types_apiece = {
        "info_type": lambda x, y, z: get_scores_info_type(y, z, n),
        "precision": lambda x, y, z: get_scores_average(x, default_score=0.0),
        "ap": lambda x, y, z: get_absolute_scores_ap(x, n),
        # "ndcg": lambda x, y, z: get_absolute_scores_ndcg(x, n),
        # "precision_length_adjusted": lambda x, y, z: get_absolute_scores_precision_length_adjusted(x),
        # "rr": lambda x, y, z: get_absolute_scores_rr(x, n),
    }
    score_types_joint = {
        # "info_type": lambda x, y, z: get_scores_info_type(y, z, n),
        "win-rate": lambda x, y, z: get_scores_average(x, default_score=0.5), ### essentially win-rate of method A over method B
    }

    dataset_keys = [
        # "test_full_10",
        # "test_segment_10",
        
        "custom_full_50",
        # "cross_full_50",
        
        # "cross_segment_50",
    ]
    method_keys = [
        "ours_similarity_openai_t05_vlatest",
        # "ours_distance_openai_t05_vlatest",

        "RAG_tfidf_k2_gpt-5-mini",
        "RAG_tfidf_k4_gpt-5-mini", ### best wrt MAP & precision among RAG
        "RAG_tfidf_k8_gpt-5-mini",
        "RAG_tfidf_k16_gpt-5-mini",
        # "RAG_tfidf_k32_gpt-5-mini",

        "RAG_tfidf_k2_gpt-4-1-mini",
        "RAG_tfidf_k4_gpt-4-1-mini",
        "RAG_tfidf_k8_gpt-4-1-mini",  ### best info-type and precision among RAG-gpt-4-1
        "RAG_tfidf_k16_gpt-4-1-mini", ### best MAP among RAG-gpt-4-1
        # "RAG_tfidf_k32_gpt-4-1-mini", ### best MAP & precision among RAG-gpt-4-1
        
        # "vanilla_tfidf_gpt-4-1-mini",
        # "vanilla_tfidf_gpt-5-mini",
    ]
    
    abs_eval_keys = [
        "relevance_absolute_gpt-5-mini_binary_n",
        # "relevance_absolute_gpt-5-mini_likert-3_n",
        # "relevance_absolute_gpt-5-mini_likert-5_n",
        # "relevance_absolute_gpt-5-mini_likert-5_n5",
    ]
    abs_combinations = list(itertools.product(dataset_keys, method_keys, abs_eval_keys))
    
    ### compares the first method with all other methods
    comp_eval_keys = [
        "relevance_comparative_gpt-5-mini_comparison_n5"
    ]
    comp_combinations = list(itertools.product(dataset_keys, [(method_keys[0], other_method_key) for other_method_key in method_keys[1:]], comp_eval_keys))

    abs_combined_results = defaultdict(lambda: defaultdict(list))
    comp_combined_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for combination in abs_combinations:
        dataset_key, method_key, eval_key = combination
        dataset_config = DATASETS[dataset_key]
        method_config = METHODS[method_key]
        eval_config = EVALS[eval_key]
        cur_results = run_absolute_eval(dataset_config, method_config, eval_config)
        abs_combined_results[method_key][f"{dataset_key}_{eval_key}"] = cur_results
    

    print("Abs Evaluation Results:")
    output_results(abs_combined_results, score_types_apiece)
    print("-"*100)

    
    for combination in comp_combinations:
        dataset_key, (method_key_A, method_key_B), eval_key = combination
        dataset_config = DATASETS[dataset_key]
        method_config_A = METHODS[method_key_A]
        method_config_B = METHODS[method_key_B]
        eval_config = EVALS[eval_key]
        cur_results = run_comparative_eval(dataset_config, method_config_A, method_config_B, eval_config)
        comp_combined_results[f"{method_key_A}_{method_key_B}"][f"{dataset_key}_{eval_key}"] = cur_results
    
    print("Comp Evaluation Results:")
    output_results(comp_combined_results, score_types_joint)
    print("-"*100)

if __name__ == "__main__":
    main()