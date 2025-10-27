"""
Evaluation prompts for LLM-as-a-Judge framework
TODO: evaluate one retrieved information at a time instead of full response
"""
from helpers import get_response_pydantic
from pydantic_models.evaluation import MetricScale
from pydantic_models.evaluation import Likert7EvaluationResponse, Likert5EvaluationResponse, BinaryEvaluationResponse, ComparisonEvaluationResponse, Likert3EvaluationResponse

def get_response_for_metric(messages, metric):
    if metric == MetricScale.LIKERT_7:
        response_format = Likert7EvaluationResponse
    elif metric == MetricScale.LIKERT_5:
        response_format = Likert5EvaluationResponse
    elif metric == MetricScale.LIKERT_3:
        response_format = Likert3EvaluationResponse
    elif metric == MetricScale.BINARY:
        response_format = BinaryEvaluationResponse
    elif metric == MetricScale.COMPARISON:
        response_format = ComparisonEvaluationResponse
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    response = get_response_pydantic(messages, response_format)
    return {
        **response,
        "confidence": response["confidence"] / 10,
        "reasoning": response["reasoning"]
    }

SYSTEM_PROMPT_EVAL = """
You are a response evaluator for a query about a tutorial for a procedural task `{task}`.
"""

USER_PROMPT_EVAL_FULL_TUTORIAL = """
You are given a context tutorial, a query, and response.

<query>
{query}
</query>

<context tutorial>
{context_tutorial}
</context tutorial>

<response>
{response}
</response>

Evaluate the response based on the following criteria:
{criteria}
"""

def tutorial_to_str(tutorial):
    TUTORIAL_FORMAT = "<tutorial title={title}>\n{content}\n</tutorial>"
    return TUTORIAL_FORMAT.format(title=tutorial['title'], content=tutorial['content'])

def response_to_str(response):
    PIECE_FORMAT = "<piece idx={idx}> {content} </piece>"
    pieces_str = ""
    for idx, piece in enumerate(response):
        pieces_str += PIECE_FORMAT.format(idx=idx+1, content=piece['content']) + "\n"
    return pieces_str


def eval_relevance_absolute_full_tutorial(task, tutorial, query, response, metric, criteria):
    if metric == MetricScale.COMPARISON:
        raise ValueError("Comparison metrics are not supported for absolute evaluation.")
    context_tutorial_str = tutorial_to_str(tutorial)
    response_str = response_to_str(response)

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_EVAL.format(task=task),
        },
        {
            "role": "user",
            "content": USER_PROMPT_EVAL_FULL_TUTORIAL.format(query=query, context_tutorial=context_tutorial_str, response=response_str, criteria=criteria),
        },
    ]

    response = get_response_for_metric(messages, metric)
    return response

USER_PROMPT_EVAL_TUTORIAL_SEGMENT = """
You are given a context tutorial, a highlighted segment, a query, and response.

<query>
{query}
</query>

<context tutorial>
{context_tutorial}
</context tutorial>

<highlighted_segment>
{highlighted_segment}
</highlighted_segment>

<response>
{response}
</response>

Evaluate the response based on the following criteria:
{criteria}
"""

def eval_relevance_absolute_tutorial_segment(task, tutorial, query, segment, response, metric, criteria):
    if metric == MetricScale.COMPARISON:
        raise ValueError("Comparison metrics are not supported for absolute evaluation.")
    context_tutorial_str = tutorial_to_str(tutorial)
    response_str = response_to_str(response)

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_EVAL.format(task=task),
        },
        {
            "role": "user",
            "content": USER_PROMPT_EVAL_TUTORIAL_SEGMENT.format(query=query, context_tutorial=context_tutorial_str, highlighted_segment=segment, response=response_str, criteria=criteria),
        },
    ]

    response = get_response_for_metric(messages, metric)
    return response

def eval_relevance_comparison_full_tutorial(metric, criteria):
    if metric != MetricScale.COMPARISON:
        raise ValueError("Metric must be comparison for comparison evaluation.")
    ### random shuffle the order of the responses

def eval_relevance_comparison_tutorial_segment(metric, criteria):
    if metric != MetricScale.COMPARISON:
        raise ValueError("Metric must be comparison for comparison evaluation.")
    pass

def eval_comprehensiveness_comparison_full_tutorial(metric, criteria):
    pass

def eval_comprehensiveness_comparison_tutorial_segment(metric, criteria):
    pass