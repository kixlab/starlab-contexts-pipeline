"""
Simplified LLM-as-a-Judge Framework

A streamlined framework for evaluating responses using LLM judges with support for:
- Binary decisions (pass/fail, relevant/irrelevant)
- Likert scale ratings (1-7, 1-5)
- Head-to-head comparisons (which is better)
"""

import json
import os

from prompts.evaluation import (
    eval_relevance_absolute_full_tutorial, eval_relevance_absolute_tutorial_segment, eval_relevance_comparison_full_tutorial, eval_relevance_comparison_tutorial_segment, eval_comprehensiveness_comparison_full_tutorial, eval_comprehensiveness_comparison_tutorial_segment
)

from pydantic_models.evaluation import MetricScale

RELEVANCE_CRITERIA_LIKERT_3 = """
Provide a score between 1 and 3 with the following meanings:
3: Highly relevant and helpful information.
2: Relevant, but not helpful.
1: Not relevant or present in the context tutorial..

Assume that you are trying to learn based on the given context tutorial. If the information in the response is already present in the context tutorial, then give a score of 0. Otherwise, if having the information in the response is crucial for learning, then give a score of 3. If you would use the information in the response to learn, then give a score of 2. If you would not use the information in the response to learn, then give a score of 1.
"""

RELEVANCE_CRITERIA_BINARY = """
Identify if the information in the response are relevant to the query or not:
yes: The information in the response are relevant to the query and is missing from the context tutorial.
no: The information in the response are not relevant to the query or is present in the context tutorial.

Assume that you are trying to learn based on the given context tutorial. If the information in the response is relevant to the query and is missing from the context tutorial, then say yes, otherwise say no.
"""

RELEVANCE_CRITERIA_COMPARISON = """
"""

RELEVANCE_CRITERIA_LIKERT_5 = """
"""

COMPREHENSIVENESS_CRITERIA_COMPARISON = """
"""

def relevance_absolute_evaluation(responses, test_dataset, metric):
    criteria = ""
    if metric == MetricScale.LIKERT_3:
        criteria = RELEVANCE_CRITERIA_LIKERT_3
    elif metric == MetricScale.LIKERT_5:
        criteria = RELEVANCE_CRITERIA_LIKERT_5
    elif metric == MetricScale.BINARY:
        criteria = RELEVANCE_CRITERIA_BINARY
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    
    results = []
    for response, test_case in zip(responses, test_dataset):
        if test_case["segment"] is not None:
            result = eval_relevance_absolute_tutorial_segment(test_case["task"], test_case["tutorial"], test_case["query"], test_case["segment"], response, metric, criteria)
        else:
            result = eval_relevance_absolute_full_tutorial(test_case["task"], test_case["tutorial"], test_case["query"], response, metric, criteria)
        results.append(result)
    print("Aggregated results:")
    print(json.dumps(aggregate_results(results, metric), indent=4))
    print("-"*20)
    return results

def faithfulness_absolute_evaluation(responses, test_dataset, eval_config):
    ### likely different approach (cosine similarity with the source texts?)
    pass

def relevance_comparative_evaluation(responses_A, responses_B, test_dataset, eval_config):
    pass

def comprehensiveness_comparative_evaluation(responses_A, responses_B, test_dataset, eval_config):
    pass


def aggregate_results(results, metric):
    ### return average decision/rating and average confidence
    aggregated_result = None
    if metric == MetricScale.COMPARISON:
        aggregated_result = aggregate_results_comparison(results)
    else:
        aggregated_result = aggregate_results_absolute(results)
    return {
        **aggregated_result,
        "metric": metric,
    }

def aggregate_results_comparison(results):
    a_win = 0
    b_win = 0
    a_win_confidence = 0
    b_win_confidence = 0
    for result in results:
        if "decision" in result and result["decision"] == "A":
            a_win += 1
            a_win_confidence += result["confidence"]
        else:
            b_win += 1
            b_win_confidence += result["confidence"]
    return {
        "a_win_rate": a_win / len(results),
        "a_win_confidence": a_win_confidence / a_win,
        "b_win_rate": b_win / len(results),
        "b_win_confidence": b_win_confidence / b_win,
    }

def aggregate_results_absolute(results):
    average_score = 0
    average_confidence = 0
    for result in results:
        if "decision" in result and result["decision"] == "yes":
            average_score += 1
        if "rating" in result:
            average_score += result["rating"]
        average_confidence += result["confidence"]
    return {
        "average_score": average_score / len(results),
        "average_confidence": average_confidence / len(results),
    }