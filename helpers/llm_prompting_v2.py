import json

from PIL import Image
from openai import OpenAI

from helpers import META_TITLE, ALIGNMENT_DEFINITIONS

from pydantic_models.segmentation import TaskGraph, IndexVideoSegmentation, get_segmentation_schema
from pydantic_models.summarization import MetaSummarySchema, SubgoalSummarySchema

from pydantic_models.comparison import AlignmentsSchema, AlignmentClassificationSchema, AlignmentHooksSchema

from helpers import get_response_pydantic, encode_image


def __extend_contents(contents, include_images=False):
    extended_contents = []
    for content in contents:
        extended_contents.append({
            "type": "text",
            "text": f"{content['text']}"
        })
        if include_images:
            for frame_path in content["frame_paths"]:
                frame_base64 = encode_image(frame_path)
                extended_contents.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{frame_base64}"}
                })
    return extended_contents

def __extend_subgoals(subgoals):
    message = ""
    for subgoal_idx, subgoal in enumerate(subgoals):
        message += f"- **Subgoal {subgoal_idx + 1}**:\n"
        for k, v in subgoal.items():
            key = k.capitalize()
            value = v if isinstance(v, str) else ", ".join(v)
            message += f"\t- {key}: {value}\n"
    return message

def define_common_subgoals_v2(contents, task):
    messages = [
        {"role": "system", "content": "You are a helpful assistant specializing in analyzing and extracting subgoals for task `{task}`. You are given a transcript of a how-to video and asked to define a task graph that consists of subgoals of the demonstrated procedure and dependencies between the subgoals. Ensure that the subgoals are (1) based on meaningful intermediate stages of the procedure, (2) broad enough to encompass diverse ways to complete the task, and (3) specific enough to capture all critical procedural steps.".format(task=task)},
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"Contents:\n"
            }] + __extend_contents(contents),
        },
    ]
    
    response = get_response_pydantic(messages, TaskGraph)
    subgoals = response["subgoals"]
    return subgoals

def align_common_subgoals_v2(common_subgoals_per_video, task):
    ### TODO: try some clustering methods to aggregate the subgoals
    messages = [
        {"role": "system", "content": "You are a helpful assistant specializing in analyzing and comparing procedural content across different how-to videos about task `{task}`. Given the task graphs (i.e., subgoals and dependencies between them) from multiple how-to videos about the same task, combine them into a single task graph. Where necessary, merge/split subgoals, and generate a unified, comprehensive set of subgoals that is applicable to all the videos. Ensure that the subgoals are based on meaningful intermediate stages of the procedure.".format(task=task)},
    ]
    for video_idx, (_, common_subgoals) in enumerate(common_subgoals_per_video.items()):
        messages.append({"role": "user", "content": f"## Video {video_idx + 1}:\n{__extend_subgoals(common_subgoals)}"})
    
    response = get_response_pydantic(messages, TaskGraph)
    subgoals = response["subgoals"]
    return subgoals

def generate_common_subgoals_v3(contents, subgoals, task):
    messages = [
        {"role": "system", "content": "You are a helpful assistant specializing in analyzing and segmenting how-to videos according to task graph for task `{task}`. Given the contents of the how-to video and subgoals, segment the video according to subgoals.".format(task=task)},
    ]

    messages.append({
        "role": "user",
        "content": f"## Subgoals:\n{__extend_subgoals(subgoals)}"
    })

    messages.append({
        "role": "user",
        "content": [{
            "type": "text",
            "text": f"Contents:\n"
        }] + __extend_contents(contents),
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
                continue

        common_subgoals.append({
            "title": content["title"],
            "explanation": content["explanation"],
            "start": content["start"],
            "finish": content["finish"],
            "text": content["text"],
            "frame_paths": content["frame_paths"],
        })
    return common_subgoals

def get_meta_summary_v2(contents, task):
    messages = [
        {"role": "system", "content": "You are a helpful assistant specializing in analyzing and summarizing procedural contents of how-to videos for task `{task}`. Given the contents from a how-to video (i.e., narration and images), extract and summarize the overall context (i.e., goal and outcome) and approach (steps). Your summaries are concise, focused, and avoid unnecessary details, ensuring that the essential information is captured without redundancy.".format(task=task)},
    ]

    messages.append({
        "role": "user",
        "content": __extend_contents(contents),
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
            }] + __extend_contents(context) + [{
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
        }] + __extend_contents(contents),
    })
    response = get_response_pydantic(messages, SubgoalSummarySchema)

    response["frame_paths"] = []
    for content in contents:
        response["frame_paths"] = response["frame_paths"] + content["frame_paths"]

    return response


### TODO: May need to add images and try again!!
def get_subgoal_alignments_v2(contents1, contents2, subgoal, task):
    messages = [
        {
            "role": "system",
            "content": [{
                "type": "text",
                "text": "You are a detailed-oriented assistant specializing in analyzing and comparing procedural content across different how-to videos about the same task. You are given procedural information from two videos (current and previous) about task `{task}. Given the information for the subgoal `{subgoal}`, analyze and compare the information from each video and identify all the `new` subgoal information in the current video compared to the previous video.".format(task=task, subgoal=subgoal)
            }],
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"## Previouse Video:`\n"
            }] + __extend_contents(contents2, False),
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"## Current Video:`\n"
            }] + __extend_contents(contents1, False),
        }
    ]

    response = get_response_pydantic(messages, AlignmentsSchema)
    return response["alignments"]

def get_meta_alignments_v2(contents1, contents2, task):
    messages = [
        {
            "role": "system",
            "content": [{
                "type": "text",
                "text": "You are a detailed-oriented assistant specializing in analyzing and comparing procedural content across different how-to videos about the same task. You are given procedural information from two videos (current and previous) about task `{task}. Analyze and compare the information from each video and identify all the `new` information in the current video compared to the previous video.".format(task=task)
            }],
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"## Previouse Video:`\n"
            }] + __extend_contents(contents2, False),
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"## Current Video:`\n"
            }] + __extend_contents(contents1, False),
        }
    ]

    response = get_response_pydantic(messages, AlignmentsSchema)
    return response["alignments"]

# def generate_common_subgoals_v2(contents, subgoals, task):
#     messages = [
#         {"role": "system", "content": "You are a helpful assistant specializing in analyzing and segmenting how-to videos according to task graph for {task}. Given the contents of the how-to video and subgoals, segment the video according to subgoals. If some subtitles do not belong to any subgoal, define custom subgoals & label them as `(custom)`.".format(task=task)},
#     ]
#     messages.append({"role": "user", "content": "Subgoals:\n"})
#     for subgoal in subgoals:
#         message = "Subgoal: {title}\nDefinition: {definition} {explanation}\nDependencies: {dependencies}".format(
#             title=subgoal["title"],
#             definition=subgoal["definition"],
#             explanation=subgoal["explanation"],
#             dependencies="; ".join(subgoal["dependencies"]),
#         )
#         messages[-1]["content"] += message + "\n"
    
#     for content in contents:
#         id = content["id"]
#         text = content["text"]
#         messages.append({
#             "role": "user",
#             "content": [{"type": "text", "text": f"{id}:\n```{text}```"}],
#         })

#         # for frame_path in content["frame_paths"]:
#         #     frame_base64 = encode_image(frame_path)
#         #     messages[-1]["content"].append({
#         #         "type": "image_url",
#         #         "image_url": {"url": f"data:image/jpeg;base64,{frame_base64}"}
#         #     })
    
#     response = get_response_pydantic(messages, IndexVideoSegmentation)
#     common_subgoals = response["segments"]

#     ### todo: find the frame where result is depicted for each subgoal
#     segment_to_subgoal = {}
#     for subgoal in common_subgoals:
#         for content_id in subgoal["content_ids"]:
#             segment_to_subgoal[content_id] = {
#                 "title": subgoal["title"],
#                 "explanation": subgoal["explanation"]
#             }
    
#     result = []
#     for content in contents:
#         title = "Custom"
#         explanation = "Custom subgoal"
#         if content["id"] in segment_to_subgoal:
#             title = segment_to_subgoal[content["id"]]["title"]
#             explanation = segment_to_subgoal[content["id"]]["explanation"]

#         if len(result) > 0 and result[-1]["title"] == title:
#             result[-1]["finish"] = content["finish"]
#             result[-1]["text"] += content["text"] + f" ({float_to_str(content['finish'])}) "
#             result[-1]["frame_paths"] = result[-1]["frame_paths"] + content["frame_paths"]
#             result[-1]["contents"].append(content)
#         else:
#             result.append({
#                 "start": content["start"],
#                 "finish": content["finish"],
#                 "text": f"({float_to_str(content['start'])}) " + content["text"] + f" ({float_to_str(content['finish'])}) ",
#                 "frame_paths": content["frame_paths"],
#                 "title": title,
#                 "explanation": explanation,
#                 "contents": [content],
#             })
#     return result