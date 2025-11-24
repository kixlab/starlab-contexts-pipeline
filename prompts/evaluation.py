"""
Evaluation prompts for LLM-as-a-Judge framework
"""
from prompts import tutorial_to_str, response_to_str
from helpers.dataset import IMPORTANT_TYPE_DESCRIPTIONS_FINE

SYSTEM_PROMPT_EVAL = """
You are a helpful assistant that carefully and objectively evaluates query responses about a procedural task `{task}` based on the given criteria."""

USER_PROMPT_EVAL_JOINT_FULL_TUTORIAL = """
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

def eval_joint_absolute_full_tutorial_request(task, tutorial, query, eval_response, criteria, response_format, judge_model):
    cur_tutorial_str = tutorial_to_str(tutorial)
    response_str = response_to_str(eval_response)

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_EVAL.format(task=task),
        },
        {
            "role": "user",
            "content": USER_PROMPT_EVAL_JOINT_FULL_TUTORIAL.format( cur_tutorial=cur_tutorial_str, query=query, response=response_str, criteria=criteria),
        },
    ]

    return {
        "messages": messages,
        "model": judge_model,
        "response_format": response_format,
    }

USER_PROMPT_EVAL_JOINT_TUTORIAL_SEGMENT = """
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

def eval_joint_absolute_tutorial_segment_request(task, tutorial, segment, query, eval_response, criteria, response_format, judge_model):
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
            "content": USER_PROMPT_EVAL_JOINT_TUTORIAL_SEGMENT.format(cur_tutorial=cur_tutorial_str, cur_segment=cur_segment_str, query=query, response=response_str, criteria=criteria),
        },
    ]

    return {
        "messages": messages,
        "model": judge_model,
        "response_format": response_format,
    }

def eval_joint_absolute_request(task, tutorial, segment, query, eval_response, criteria, response_format, judge_model, **kwargs):
    if segment is None:
        return eval_joint_absolute_full_tutorial_request(task, tutorial, query, eval_response, criteria, response_format, judge_model)
    else:
        return eval_joint_absolute_tutorial_segment_request(task, tutorial, segment, query, eval_response, criteria, response_format, judge_model)

USER_PROMPT_EVAL_APIECE_FULL_TUTORIAL = """
User requested additional information of type `{info_type}` in the context of the current tutorial and you are given a response to the request. The `{info_type}` is information type defined as follows:
{info_type_definition}

Evaluate the response based on the following criteria:
{criteria}

Current tutorial:
{cur_tutorial}

Response:
{response}
"""

def eval_apiece_absolute_full_tutorial_request(task, tutorial, info_type, eval_response, criteria, response_format, judge_model):
    cur_tutorial_str = tutorial_to_str(tutorial)
    response_str = response_to_str([eval_response])
    info_type_definition = IMPORTANT_TYPE_DESCRIPTIONS_FINE[info_type].strip()

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_EVAL.format(task=task),
        },
        {
            "role": "user",
            "content": USER_PROMPT_EVAL_APIECE_FULL_TUTORIAL.format(
                info_type=info_type,
                info_type_definition=info_type_definition,
                criteria=criteria,
                cur_tutorial=cur_tutorial_str,
                response=response_str,
            ),
        },
    ]

    return {
        "messages": messages,
        "model": judge_model,
        "response_format": response_format,
    }

USER_PROMPT_EVAL_APIECE_TUTORIAL_SEGMENT = """
User requested additional information of type `{info_type}` in the context of the current tutorial and its segment. The `{info_type}` is information type defined as follows:
{info_type_definition}

Evaluate the response based on the following criteria:
{criteria}

Current tutorial:
{cur_tutorial}

Current segment:
{cur_segment}

Response:
{response}
"""

def eval_apiece_absolute_tutorial_segment_request(task, tutorial, segment, info_type, eval_response, criteria, response_format, judge_model):
    cur_tutorial_str = tutorial_to_str(tutorial)
    cur_segment_str = segment["content"]
    response_str = response_to_str([eval_response])
    info_type_definition = IMPORTANT_TYPE_DESCRIPTIONS_FINE[info_type].strip()

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_EVAL.format(task=task),
        },
        {
            "role": "user",
            "content": USER_PROMPT_EVAL_APIECE_TUTORIAL_SEGMENT.format(
                info_type=info_type,
                info_type_definition=info_type_definition,
                criteria=criteria,
                cur_tutorial=cur_tutorial_str,
                cur_segment=cur_segment_str,
                response=response_str,
            ),
        },
    ]

    return {
        "messages": messages,
        "model": judge_model,
        "response_format": response_format,
    }

def eval_apiece_absolute_request(task, tutorial, segment, info_type, eval_response, criteria, response_format, judge_model, **kwargs):
    if segment is None:
        return eval_apiece_absolute_full_tutorial_request(task, tutorial, info_type, eval_response, criteria, response_format, judge_model)
    else:
        return eval_apiece_absolute_tutorial_segment_request(task, tutorial, segment, info_type, eval_response, criteria, response_format, judge_model)

def eval_absolute_response(response, max_rating, **kwargs):
    rating = 0
    if "rating" in response:
        rating = (response["rating"] - 1) / (max_rating - 1)
    elif "decision" in response:
        if response["decision"] == "yes":
            rating = 1
        else:
            rating = 0
    return {
        "rating": rating,
        "confidence": (response["confidence"] - 1) / (10 - 1),
        "reasoning": response["reasoning"]
    }
