"""
Evaluation prompts for LLM-as-a-Judge framework
"""
from prompts import tutorial_to_str, response_to_str

SYSTEM_PROMPT_EVAL = """
You are a helpful assistant that carefully and objectively evaluates the responses to the query about a procedural task `{task}` based on the given criteria."""

USER_PROMPT_EVAL_FULL_TUTORIAL = """
You are given a query and a response to the query. The query was originally asked in the context of the current tutorial.
Evaluate the response based on the following criteria:
{criteria}

Current tutorial:
{cur_tutorial}

Original query:
{query}

Response:
{response}
"""

def _eval_absolute_full_tutorial_request(task, tutorial, query, eval_response, criteria, response_format, judge_model):
    cur_tutorial_str = tutorial_to_str(tutorial)
    response_str = response_to_str(eval_response)

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_EVAL.format(task=task),
        },
        {
            "role": "user",
            "content": USER_PROMPT_EVAL_FULL_TUTORIAL.format( cur_tutorial=cur_tutorial_str, query=query, response=response_str, criteria=criteria),
        },
    ]

    return {
        "messages": messages,
        "model": judge_model,
        "response_format": response_format,
    }

USER_PROMPT_EVAL_TUTORIAL_SEGMENT = """
You are given a query and a response to the query. The query was originally asked in the context of the current tutorial and its segment.
Evaluate the response based on the following criteria:
{criteria}

Current tutorial:
{cur_tutorial}

Current segment:
{cur_segment}

Original query:
{query}

Response:
{response}
"""

def _eval_absolute_tutorial_segment_request(task, tutorial, segment, query, eval_response, criteria, response_format, judge_model):
    cur_tutorial_str = tutorial_to_str(tutorial)
    cur_segment_str = segment["content"]
    response_str = response_to_str(eval_response)

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_EVAL.format(task=task),
        },
        {
            "role": "user",
            "content": USER_PROMPT_EVAL_TUTORIAL_SEGMENT.format(cur_tutorial=cur_tutorial_str, cur_segment=cur_segment_str, query=query, response=response_str, criteria=criteria),
        },
    ]

    return {
        "messages": messages,
        "model": judge_model,
        "response_format": response_format,
    }

def eval_absolute_request(task, tutorial, segment, query, eval_response, criteria, response_format, judge_model, **kwargs):
    if segment is None:
        return _eval_absolute_full_tutorial_request(task, tutorial, query, eval_response, criteria, response_format, judge_model)
    else:
        return _eval_absolute_tutorial_segment_request(task, tutorial, segment, query, eval_response, criteria, response_format, judge_model)

def eval_absolute_response(response, max_rating, **kwargs):
    rating = None
    if "rating" in response:
        rating = (response["rating"] - 1) / (max_rating - 1)
    elif "decision" in response:
        if response["decision"] == "yes":
            rating = 1
        else:
            rating = 0
    if rating is None:
        return None
    return {
        "rating": rating,
        "confidence": (response["confidence"] - 1) / (10 - 1),
        "reasoning": response["reasoning"]
    }

USER_PROMPT_EVAL_COMPARATIVE_FULL_TUTORIAL = """
You are given a query and two responses to the query. The query was originally asked in the context of the current tutorial.
Evaluate the responses based on following criteria:
{criteria}

Current tutorial:
{cur_tutorial}

Original query:
{query}

Response A:
{response_A}

Response B:
{response_B}
"""

def _eval_comparative_full_tutorial_request(task, tutorial, query, eval_response_A, eval_response_B, criteria, response_format, judge_model):
    cur_tutorial_str = tutorial_to_str(tutorial)
    response_str_A = response_to_str(eval_response_A)
    response_str_B = response_to_str(eval_response_B)

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_EVAL.format(task=task),
        },
        {
            "role": "user",
            "content": USER_PROMPT_EVAL_COMPARATIVE_FULL_TUTORIAL.format(cur_tutorial=cur_tutorial_str, query=query, response_A=response_str_A, response_B=response_str_B, criteria=criteria),
        },
    ]

    return {
        "messages": messages,
        "model": judge_model,
        "response_format": response_format,
    }


USER_PROMPT_EVAL_COMPARATIVE_TUTORIAL_SEGMENT = """
You are given a query and two responses to the query. The query was originally asked in the context of the current tutorial and its segment.
Evaluate the responses based on following criteria:
{criteria}

Current tutorial:
{cur_tutorial}

Current segment:
{cur_segment}

Original query:
{query}

Response A:
{response_A}

Response B:
{response_B}
"""

def _eval_comparative_tutorial_segment_request(task, tutorial, segment, query, eval_response_A, eval_response_B, criteria, response_format, judge_model):
    cur_tutorial_str = tutorial_to_str(tutorial)
    cur_segment_str = segment["content"]
    response_str_A = response_to_str(eval_response_A)
    response_str_B = response_to_str(eval_response_B)

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_EVAL.format(task=task),
        },
        {
            "role": "user",
            "content": USER_PROMPT_EVAL_COMPARATIVE_TUTORIAL_SEGMENT.format(cur_tutorial=cur_tutorial_str, cur_segment=cur_segment_str, query=query, response_A=response_str_A, response_B=response_str_B, criteria=criteria),
        },
    ]

    return {
        "messages": messages,
        "model": judge_model,
        "response_format": response_format,
    }

def eval_comparative_request(task, tutorial, segment, query, eval_response_A, eval_response_B, criteria, response_format, judge_model, **kwargs):
    if segment is None:
        return _eval_comparative_full_tutorial_request(task, tutorial, query, eval_response_A, eval_response_B, criteria, response_format, judge_model)
    else:
        return _eval_comparative_tutorial_segment_request(task, tutorial, segment, query, eval_response_A, eval_response_B, criteria, response_format, judge_model)

def eval_comparative_response(response, **kwargs):
    rating = None
    if response["decision"] == "A":
        rating = 1
    if response["decision"] == "B":
        rating = 0
    if response["decision"] == "tie":
        rating = 0.5
    if rating is None:
        return None
    return {
        "rating": rating,
        "confidence": (response["confidence"] - 1) / (10 - 1),
        "reasoning": response["reasoning"]
    }