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
    eval_joint_absolute_request,
    eval_apiece_absolute_request,
    eval_absolute_response,
)

from pydantic_models.evaluation import MetricScale
from pydantic_models.evaluation import Likert3EvaluationResponse, Likert5EvaluationResponse, BinaryEvaluationResponse


from prompts.framework_batch import batch_run_lm_calls

from prompts.metrics_criteria import (
    RELEVANCE_CRITERIA_LIKERT_3,
    RELEVANCE_CRITERIA_LIKERT_5,
    RELEVANCE_CRITERIA_BINARY,
)

def _apiece_absolute_evaluation(eval_responses, test_dataset, criteria, max_rating, response_format, judge_model):
    request_args = []
    req_idx_to_source = []
    for i, (eval_response, test_case) in enumerate(zip(eval_responses, test_dataset)):
        if eval_response is None or len(eval_response) == 0:
            continue
        task = test_case["task"]
        tutorial = test_case["tutorial"]
        segment = test_case["segment"]
        info_type = test_case["info_type"]
        for eval_response_i in eval_response:
            request_args.append({
                "task": task,
                "tutorial": tutorial,
                "segment": segment,
                "info_type": info_type,
                "eval_response": eval_response_i,
                "response_format": response_format,
                "criteria": criteria,
                "judge_model": judge_model,
                "max_rating": max_rating,
            })
            req_idx_to_source.append(i)

    batch_results = batch_run_lm_calls(request_args, eval_apiece_absolute_request, eval_absolute_response)

    results = []
    for i in range(len(test_dataset)):
        results.append([])
    for result, test_idx in zip(batch_results, req_idx_to_source):
        results[test_idx].append(result)
    return results

def _joint_absolute_evaluation(eval_responses, test_dataset, criteria, max_rating, response_format, judge_model):
    request_args = []
    req_idx_to_source = []
    for i, (eval_response, test_case) in enumerate(zip(eval_responses, test_dataset)):
        if eval_response is None or len(eval_response) == 0:
            continue
        task = test_case["task"]
        tutorial = test_case["tutorial"]
        segment = test_case["segment"]
        query = test_case["query"]

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

    batch_results = batch_run_lm_calls(request_args, eval_joint_absolute_request, eval_absolute_response)

    results = []
    for i in range(len(test_dataset)):
        results.append([])
    for result, test_idx in zip(batch_results, req_idx_to_source):
        results[test_idx].append(result)
    return results

def relevance_absolute_evaluation(eval_responses, test_dataset, metric, joint, judge_model):
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

    if joint:
        return _joint_absolute_evaluation(eval_responses, test_dataset, criteria, max_rating, response_format, judge_model)
    else:
        return _apiece_absolute_evaluation(eval_responses, test_dataset, criteria, max_rating, response_format, judge_model)


def get_absolute_scores_average(results):
    scores = []
    for result in results:
        if len(result) == 0:
            scores.append(0.0) ## minimum score
        else:
            scores.append(np.mean([r["rating"] for r in result]))
    return scores

def get_absolute_scores_precision(results, k):
    scores = []
    for result in results:
        if len(result) == 0:
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
        if len(result) == 0:
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
            scores.append(average_precision / k)
    return scores

def get_absolute_scores_ndcg(results, k):
    ideal_dcg = 0
    for i in range(k):
        ideal_dcg += 1 / np.log2(i + 2)

    scores = []
    for result in results:
        if len(result) == 0:
            scores.append(0.0) ## minimum score
        else:
            dcg = 0
            for i in range(k):
                if i >= len(result):
                    break
                dcg += result[i]["rating"] / np.log2(i + 2)
            scores.append(dcg / ideal_dcg)
    return scores