from helpers import get_response_pydantic, extend_contents, extend_subgoals
from helpers import encode_image, get_response_pydantic_with_message

from pydantic_models.segmentation import TaskGraph, get_segmentation_schema, ListSubgoalsSchema, SubgoalSchema, AllProceduralInformationSchema


def define_common_subgoals_v2(contents, task):
    messages = [
        {"role": "system", "content": "You are a helpful assistant specializing in analyzing and extracting subgoals for task `{task}`. You are given a transcript of a how-to video and asked to define a task graph that consists of subgoals of the demonstrated procedure and dependencies between the subgoals. Ensure that the subgoals are (1) based on meaningful intermediate stages of the procedure, (2) broad enough to encompass diverse ways to complete the task, and (3) specific enough to capture all critical procedural steps.".format(task=task)},
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"Contents:\n"
            }] + extend_contents(contents),
        },
    ]
    
    response = get_response_pydantic(messages, TaskGraph)
    subgoals = response["subgoals"]
    return subgoals

def align_common_subgoals_v2(common_subgoals_per_video, task):
    messages = [
        {"role": "system", "content": "You are a helpful assistant specializing in analyzing and comparing procedural content across different how-to videos about task `{task}`. Given the task graphs (i.e., subgoals and dependencies between them) from multiple how-to videos about the same task, combine them into a single task graph. Where necessary, merge/split subgoals, and generate a unified, comprehensive set of subgoals that is applicable to all the videos. Ensure that the subgoals are based on meaningful intermediate stages of the procedure.".format(task=task)},
    ]
    for video_idx, (_, common_subgoals) in enumerate(common_subgoals_per_video.items()):
        messages.append({"role": "user", "content": f"## Video {video_idx + 1}:\n{extend_subgoals(common_subgoals)}"})
    
    response = get_response_pydantic(messages, TaskGraph)
    subgoals = response["subgoals"]
    return subgoals

def generate_common_subgoals_v3(contents, subgoals, task):
    messages = [
        {"role": "system", "content": "You are a helpful assistant specializing in analyzing and segmenting how-to videos according to task graph for task `{task}`. Given the contents of the how-to video and subgoals, segment the video according to subgoals.".format(task=task)},
    ]

    messages.append({
        "role": "user",
        "content": f"## Subgoals:\n{extend_subgoals(subgoals)}"
    })

    messages.append({
        "role": "user",
        "content": [{
            "type": "text",
            "text": f"Contents:\n"
        }] + extend_contents(contents),
    })

    titles = [subgoal["title"] for subgoal in subgoals] + ["Custom"]
    SegmentationSchema = get_segmentation_schema(titles)

    response = get_response_pydantic(messages, SegmentationSchema)
    segmentation = response["segments"]

    for segment in segmentation:
        quotes = " ".join(segment["quotes"])
        for content in contents:
            if content["text"] in quotes or quotes in content["text"]:
                content["title"] = segment["title"]
                content["explanation"] = segment["explanation"]

    common_subgoals = []
    for content in contents:
        if "title" not in content:
            content["title"] = "Custom"
            content["explanation"] = "Custom subgoal"
        if (len(common_subgoals) > 0):
            if common_subgoals[-1]["title"] == content["title"] or content["title"] == "Custom":
                common_subgoals[-1]["finish"] = content["finish"]
                common_subgoals[-1]["text"] += " " + content["text"]
                common_subgoals[-1]["frame_paths"] = common_subgoals[-1]["frame_paths"] + content["frame_paths"]
                common_subgoals[-1]["content_ids"].append(content["id"])
                continue

        common_subgoals.append({
            "title": content["title"],
            "explanation": content["explanation"],
            "start": content["start"],
            "finish": content["finish"],
            "text": content["text"],
            "frame_paths": content["frame_paths"],
            "content_ids": [content["id"]]
        })
    return common_subgoals


def define_common_subgoals_v4(contents, task):
    messages = [
        {"role": "system", "content": "You are a helpful assistant specializing in analyzing tutorial video content. You are given a narration of a video for a task `{task}` and asked to specify a sequence of presented subgoals and their expected outcome. Ensure that subgoals encompass all procedural information presented in the video.".format(task=task)},
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"Contents:\n"
            }] + extend_contents(contents),
        },
    ]
    
    response = get_response_pydantic(messages, ListSubgoalsSchema)
    subgoals = response["subgoals"]
    return subgoals

def summarize_common_subgoals_v4(subgoals, task):
    messages = [
        {"role": "system", "content": "You are a helpful assistant specializing in analyzing tutorial content. You are given a set of similar subgoals in the task `{task}`. Summarize given subgoals into a single COMPREHENSIVE subgoal.".format(task=task)},
        {
            "role": "user",
            "content": f"## Subgoals:\n{extend_subgoals(subgoals)}"
        }
    ]
    
    response = get_response_pydantic(messages, SubgoalSchema)
    subgoal = response
    return subgoal

def extract_all_procedural_info_v5(contents, task, include_image=False):
    # extract_all_procedural_info_v5_explicit(contents, task, include_image)
    extract_all_procedural_info_v5_implicit(contents, task, include_image)

def extract_all_procedural_info_v5_implicit(contents, task, include_image=False):
    messages = [
        {"role": "system", "content": "You are a helpful assistant specializing in analyzing tutorial video content. You are given a narration of a video about the task `{task}` and asked to extract all the provided procedural information relevant to the task from each sentence.".format(task=task)},
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"Contents:\n"
            }] + extend_contents(contents, include_image),
        },
    ]

    all_pieces = []

    for content in contents:
        cur_message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Extract procedural information from the following contents:\n"
                },
                {
                    "type": "text",
                    "text": f"{content['text']}\n"
                }
            ]
        }
        if include_image:
            for frame_path in content["frame_paths"]:
                frame_base64 = encode_image(frame_path)
                cur_message["content"].append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{frame_base64}"}
                })
        messages.append(cur_message)
        response, message = get_response_pydantic_with_message(messages, AllProceduralInformationSchema)
        all_pieces.append({
            "id": content["id"],
            "pieces": response["all"],
        })
        messages.append({
            "role": "assistant",
            "content": message
        })
    return all_pieces

def extract_all_procedural_info_v5_explicit(contents, task, include_image=False):
    messages = [
        {"role": "system", "content": "You are a helpful assistant specializing in analyzing tutorial video content. You are given a narration of a video about the task `{task}` and asked to extract all the provided procedural information relevant to the task from each sentence.".format(task=task)},
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"Contents:\n"
            }] + extend_contents(contents),
        },
    ]

    all_pieces = []

    for content in contents:
        cur_message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Extract procedural information from the following contents:\n"
                },
                {
                    "type": "text",
                    "text": f"{content['text']}\n"
                }
            ]
        }
        if include_image:
            for frame_path in content["frame_paths"]:
                frame_base64 = encode_image(frame_path)
                cur_message["content"].append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{frame_base64}"}
                })
        messages.append(cur_message)
        response, message = get_response_pydantic_with_message(messages, AllProceduralInformationSchema)
        all_pieces.append({
            "id": content["id"],
            "pieces": response["all"],
        })
        messages.append({
            "role": "assistant",
            "content": message
        })
    return all_pieces