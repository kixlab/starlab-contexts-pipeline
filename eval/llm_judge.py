"""
Simplified LLM-as-a-Judge Framework

A streamlined framework for evaluating responses using LLM judges with support for:
- Binary decisions (pass/fail, relevant/irrelevant)
- Likert scale ratings (1-7, 1-5)
- Head-to-head comparisons (which is better)

References:
@inproceedings{10.1145/3626772.3657707,
author = {Thomas, Paul and Spielman, Seth and Craswell, Nick and Mitra, Bhaskar},
title = {Large Language Models can Accurately Predict Searcher Preferences},
year = {2024},
}
@article{Sebastian2025ValidatingLR,
  title={Validating LLM-Generated Relevance Labels for Educational Resource Search},
  author={Ratan J. Sebastian and Anett Hoppe},
  year={2025},
}
"""

import json
import os
import numpy as np

from prompts.evaluation import (
    eval_absolute_request,
    eval_absolute_response,
    eval_comparative_request,
    eval_comparative_response,
)

from pydantic_models.evaluation import MetricScale
from pydantic_models.evaluation import Likert3EvaluationResponse, Likert5EvaluationResponse, BinaryEvaluationResponse, ComparisonEvaluationResponse


from prompts.framework_batch import batch_run_lm_calls

from prompts.metrics_criteria import (
    RELEVANCE_CRITERIA_LIKERT_3,
    RELEVANCE_CRITERIA_LIKERT_5,
    RELEVANCE_CRITERIA_BINARY,
    RELEVANCE_CRITERIA_COMPARISON,
)

def relevance_absolute_evaluation(eval_responses, test_dataset, config):
    metric = config["metric"]
    judge_model = config["judge_model"]
    n = config["n"]

    criteria = ""
    max_rating = 0
    response_format = None
    if metric == MetricScale.LIKERT_3:
        criteria = RELEVANCE_CRITERIA_LIKERT_3
        max_rating = 3
        response_format = Likert3EvaluationResponse
    elif metric == MetricScale.LIKERT_5:
        criteria = RELEVANCE_CRITERIA_LIKERT_5
        max_rating = 5
        response_format = Likert5EvaluationResponse
    elif metric == MetricScale.BINARY:
        criteria = RELEVANCE_CRITERIA_BINARY
        max_rating = 2
        response_format = BinaryEvaluationResponse
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    request_args = []
    req_idx_to_source = []
    for i, (eval_response, test_case) in enumerate(zip(eval_responses, test_dataset)):
        if eval_response is None or len(eval_response) == 0:
            continue
        task = test_case["task"]
        tutorial = test_case["tutorial"]
        segment = test_case["segment"]
        query = test_case["query"]
        
        # info_type = test_case["info_type"]
        # ### filter out eval_response that is not of the same info_type
        # eval_response = [eval_response_i for eval_response_i in eval_response if eval_response_i["content_type"] == info_type]
        # if len(eval_response) == 0:
        #     continue

        if n is None:
            for eval_response_i in eval_response:
                request_args.append({
                    "task": task,
                    "tutorial": tutorial,
                    "segment": segment,
                    "query": query,
                    "eval_response": [eval_response_i],
                    "response_format": response_format,
                    "criteria": criteria,
                    "judge_model": judge_model,
                    "max_rating": max_rating,
                })
                req_idx_to_source.append(i)
        else:
            eval_response = eval_response[:n]
            request_args.append({
                "task": task,
                "tutorial": tutorial,
                "segment": segment,
                "query": query,
                "eval_response": eval_response,
                "response_format": response_format,
                "criteria": criteria,
                "judge_model": judge_model,
                "max_rating": max_rating,
            })
            req_idx_to_source.append(i)


    batch_results = batch_run_lm_calls(request_args, eval_absolute_request, eval_absolute_response)

    results = []
    for i in range(len(test_dataset)):
        results.append([])
    for result, test_idx in zip(batch_results, req_idx_to_source):
        if result is None:
            continue
        results[test_idx].append(result)
    return results

def relevance_comparative_evaluation(eval_responses_A, eval_responses_B, test_dataset, config):
    metric = config["metric"]
    judge_model = config["judge_model"]
    n = config["n"]

    criteria = ""
    max_rating = 0
    response_format = None
    if metric == MetricScale.COMPARISON:
        criteria = RELEVANCE_CRITERIA_COMPARISON
        max_rating = 3
        response_format = ComparisonEvaluationResponse
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    request_args = []
    req_idx_to_source = []
    for i, (eval_response_A, eval_response_B, test_case) in enumerate(zip(eval_responses_A, eval_responses_B, test_dataset)):
        task = test_case["task"]
        tutorial = test_case["tutorial"]
        segment = test_case["segment"]
        query = test_case["query"]

        if n is not None:
            eval_response_A = eval_response_A[:n]
            eval_response_B = eval_response_B[:n]

        # info_type = test_case["info_type"]
        # ### filter out eval_response_A and eval_response_B that is not of the same info_type
        # eval_response_A = [eval_response_i for eval_response_i in eval_response_A if eval_response_i["content_type"] == info_type]
        # eval_response_B = [eval_response_i for eval_response_i in eval_response_B if eval_response_i["content_type"] == info_type]

        print("Comparing eval_response_A and eval_response_B:", len(eval_response_A), len(eval_response_B))

        request_args.append({
            "task": task,
            "tutorial": tutorial,
            "segment": segment,
            "query": query,
            "eval_response_A": eval_response_A,
            "eval_response_B": eval_response_B,
            "response_format": response_format,
            "criteria": criteria,
            "judge_model": judge_model,
            "max_rating": max_rating,
        })
        req_idx_to_source.append(i)

    batch_results = batch_run_lm_calls(request_args, eval_comparative_request, eval_comparative_response)

    results = []
    for i in range(len(test_dataset)):
        results.append([])
    for result, test_idx in zip(batch_results, req_idx_to_source):
        if result is None:
            continue
        results[test_idx].append(result)
    return results

def get_scores_info_type(responses, test_dataset, k):
    scores = []
    for test_case, response in zip(test_dataset, responses):
        if response is None or len(response) == 0:
            scores.append(0.0) ## minimum score
        else:
            correct_info_type_count = 0
            for i in range(k):
                if i >= len(response):
                    break
                if response[i]["content_type"] == test_case["info_type"]:
                    correct_info_type_count += 1
            scores.append(correct_info_type_count / k)
    return scores

def get_scores_average(results, default_score=0.0):
    scores = []
    for result in results:
        if result is None or len(result) == 0:
            scores.append(default_score)
        else:
            scores.append(np.mean([r["rating"] for r in result]))
    return scores

def get_absolute_scores_precision(results, k):
    scores = []
    for result in results:
        if result is None or len(result) == 0:
            scores.append(0.0) ## minimum score
        else:
            relevant_count = 0
            for i in range(k):
                if i >= len(result):
                    break
                if result[i]["rating"] > 0.5-1e-6:
                    relevant_count += 1
            scores.append(relevant_count / k)
    return scores

def get_absolute_scores_ap(results, k):
    scores = []
    for result in results:
        if result is None or len(result) == 0:
            scores.append(0.0) ## minimum score
        else:
            average_precision = 0
            relevant_count = 0
            for i in range(k):
                if i >= len(result):
                    break
                if result[i]["rating"] > 0.5-1e-6:
                    relevant_count += 1
                    average_precision += relevant_count / (i + 1)
            scores.append(average_precision / relevant_count if relevant_count > 0 else 0.0)
    return scores

def get_absolute_scores_ndcg(results, k):
    ideal_dcg = 0
    for i in range(k):
        ideal_dcg += 1 / np.log2(i + 2)

    scores = []
    for result in results:
        if result is None or len(result) == 0:
            scores.append(0.0) ## minimum score
        else:
            dcg = 0
            for i in range(k):
                if i >= len(result):
                    break
                dcg += result[i]["rating"] / np.log2(i + 2)
            scores.append(dcg / ideal_dcg)
    return scores

def get_absolute_scores_precision_length_adjusted(results):
    scores = []
    for result in results:
        if result is None or len(result) == 0:
            scores.append(0.0) ## minimum score
        else:
            relevant_count = 0
            for i in range(len(result)):
                if result[i]["rating"] > 0.5-1e-6:
                    relevant_count += 1  
            scores.append(relevant_count / len(result))
    return scores

def get_absolute_scores_rr(results, k):
    scores = []
    for result in results:
        if result is None or len(result) == 0:
            scores.append(0.0) ## minimum score
        else:
            rr = 0
            for i in range(k):
                if i >= len(result):
                    break
                if result[i]["rating"] > 0.5-1e-6:
                    rr = 1 / (i + 1)
                    break
            scores.append(rr)
    return scores