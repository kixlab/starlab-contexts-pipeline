from helpers import get_response_pydantic, extend_contents

from pydantic_models.summarization import StepSummarySchema, SubgoalSummarySchema, StructuredSubgoalSummarySchema

def get_step_summary_v4(contents, subgoal, steps, task):
    if len(steps) == 0:
        return None

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant specializing in analyzing tutorial video content. Given a narration of a tutorial video for the task `{task}`, identify and extract all procedural information about the listed steps for `{subgoal}` based on the JSON schema.".format(task=task, subgoal=subgoal)
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"## `{subgoal}` Steps:\n" + " ".join(steps)
            }]
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"## Contents:\n"
            }] + extend_contents(contents, include_ids=True),
        }
    ]
    response = get_response_pydantic(messages, StructuredSubgoalSummarySchema)

    return augment_response(response, contents)

def get_subgoal_summary_v4(contents, subgoal, task):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant specializing in analyzing tutorial video content. Given a narration of a tutorial video for the task `{task}`, identify and extract all procedural information for subgoal `{subgoal}`.".format(task=task, subgoal=subgoal)
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"## Contents:\n"
            }] + extend_contents(contents, include_ids=True),
        }
    ]
    # response = get_response_pydantic(messages, SubgoalSummarySchema)
    response = get_response_pydantic(messages, StructuredSubgoalSummarySchema)

    return augment_response(response, contents)

def augment_response(response, contents):
    response["frame_paths"] = []
    for content in contents:
        response["frame_paths"] += [*content["frame_paths"]]        
    
    for parent_key in response:
        for obj in response[parent_key]:
            for key in obj:
                if not key.endswith("content_ids"):
                    continue
                new_content_ids = []
                for id in obj[key]:
                    new_content_ids.append(contents[id]["id"])
                obj[key] = new_content_ids
    for parent_key in response:
        if parent_key not in ["outcomes", "materials", "tools"]:
            continue
        for obj in response[parent_key]:
            obj["frame_paths"] = [obj["caption"]]
    
    ### Check non-empty
    cannot_be_empty = ["instructions", "explanations", "tips", "warnings"]
    for obj in response["steps"]:
        for key in cannot_be_empty:
            contents_key = key + "_content_ids"
            if contents_key not in obj:
                continue
            if len(obj[contents_key]) == 0:
                obj[key] = ""
    
    return response