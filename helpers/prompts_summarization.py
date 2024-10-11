from helpers import get_response_pydantic, extend_contents

from pydantic_models.summarization import MetaSummarySchema, SubgoalSummarySchema

def get_meta_summary_v2(contents, task):
    messages = [
        {"role": "system", "content": "You are a helpful assistant specializing in analyzing and summarizing procedural contents of how-to videos for task `{task}`. Given the contents from a how-to video (i.e., narration and images), extract and summarize the overall context (i.e., goal and outcome) and approach (steps). Your summaries are concise, focused, and avoid unnecessary details, ensuring that the essential information is captured without redundancy.".format(task=task)},
    ]

    messages.append({
        "role": "user",
        "content": extend_contents(contents),
    })
    response = get_response_pydantic(messages, MetaSummarySchema)

    response["frame_paths"] = []
    for content in contents:
        response["frame_paths"] = response["frame_paths"] + content["frame_paths"]
    return response

def get_subgoal_summary_v2(contents, context, subgoal, task):
    messages = [
        {
            "role": "system",
            "content": [{
                "type": "text",
                "text": "You are a helpful assistant specializing in analyzing and summarizing procedural information of subgoals in how-to videos for task `{task}`. Here is the overall summary of the video:\n".format(task=task)
            }] + extend_contents(context) + [{
                "type": "text",
                "text": "Particularly, you are given the contents about the step `{subgoal}` from this video. Extract and summarize following procedural information according to the contents: subgoal, materials, outcome, instructions, rationale, and tips.".format(subgoal=subgoal)
            }],
        }
    ]

    ### contents
    messages.append({
        "role": "user",
        "content": [{
            "type": "text",
            "text": f"Contents:\n"
        }] + extend_contents(contents),
    })
    response = get_response_pydantic(messages, SubgoalSummarySchema)

    response["frame_paths"] = []
    for content in contents:
        response["frame_paths"] = response["frame_paths"] + content["frame_paths"]

    return response

