import json

from PIL import Image
from openai import OpenAI

from helpers import META_TITLE, ALIGNMENT_DEFINITIONS

from pydantic_models.segmentation import TaskGraph, IndexVideoSegmentation, SegmentationSchema
from pydantic_models.summarization import MetaSummarySchema, SubgoalSummarySchema

from pydantic_models.comparison import AlignmentsSchema, AlignmentClassificationSchema, AlignmentHooksSchema

from helpers import get_response_pydantic, float_to_str, segment_into_sentences, encode_image



def define_common_subgoals_v2(contents, task):
    messages = [
        {"role": "system", "content": "You are a helpful assistant specializing in analyzing and extracting subgoals for {task}. You are given a transcript of a how-to video and asked to define a task graph that consists of subgoals of the demonstrated procedure and dependencies between the subgoals. Ensure that the subgoals are (1) based on meaningful intermediate stages of the procedure, (2) broad enough to encompass diverse ways to complete the task, and (3) specific enough to capture all critical procedural steps.".format(task=task)},
        {"role": "user", "content": f"Contents:\n```{' '.join([content['text'] for content in contents])}```"}
    ]
    
    subgoals = get_response_pydantic(messages, TaskGraph)
    return subgoals

def align_common_subgoals_v2(common_subgoals_per_video, task):
    ### TODO: try some clustering methods to aggregate the subgoals
    messages = [
        {"role": "system", "content": "You are a helpful assistant specializing in analyzing and comparing procedural content across different how-to videos about {task}. Given the task graphs (i.e., subgoals and dependencies between them) from multiple how-to videos about the same task, combine them into a single task graph. Where necessary, merge/split subgoals, and generate a unified, comprehensive set of subgoals that is applicable to all the videos. Ensure that the subgoals are based on meaningful intermediate stages of the procedure.".format(task=task)},
    ]
    for video_idx, (_, common_subgoals) in enumerate(common_subgoals_per_video.items()):
        message = ""
        for subgoal_idx, subgoal in enumerate(common_subgoals):
            message += f"- **Subgoal {subgoal_idx + 1}**:\n"
            for k, v in subgoal.items():
                key = k.capitalize()
                value = v if isinstance(v, str) else ", ".join(v)
                message += f"\t- {key}: {value}\n"
        messages.append({"role": "user", "content": f"Video {video_idx + 1}:\n```{message}```"})
    
    subgoals = get_response_pydantic(messages, TaskGraph)
    return subgoals

def generate_common_subgoals_v3(contents, subgoals, task):
    messages = [
        {"role": "system", "content": "You are a helpful assistant specializing in analyzing and segmenting how-to videos according to task graph for {task}. Given the contents of the how-to video and subgoals, segment the video according to subgoals. If some subtitles do not belong to any subgoal, define custom subgoals.".format(task=task)},
    ]

    ### SUBGOALS in markdown format
    subgoals_message = ""
    for subgoal_idx, subgoal in enumerate(subgoals):
        subgoals_message += f"- **Subgoal {subgoal_idx + 1}**:\n"
        for k, v in subgoal.items():
            key = k.capitalize()
            value = v if isinstance(v, str) else ", ".join(v)
            subgoals_message += f"\t- {key}: {value}\n"
    messages.append({"role": "user", "content": f"Subgoals:\n```{subgoals_message}```"})

    messages.append({"role": "user", "content": f"Contents:\n```{' '.join([content['text'] for content in contents])}```"})

    segmentation = get_response_pydantic(messages, SegmentationSchema)

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
                common_subgoals[-1]["frame_paths"].extend(content["frame_paths"])
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

def generate_common_subgoals_v2(contents, subgoals, task):
    messages = [
        {"role": "system", "content": "You are a helpful assistant specializing in analyzing and segmenting how-to videos according to task graph for {task}. Given the contents of the how-to video and subgoals, segment the video according to subgoals. If some subtitles do not belong to any subgoal, define custom subgoals & label them as `(custom)`.".format(task=task)},
    ]
    messages.append({"role": "user", "content": "Subgoals:\n"})
    for subgoal in subgoals:
        message = "Subgoal: {title}\nDefinition: {definition} {explanation}\nDependencies: {dependencies}".format(
            title=subgoal["title"],
            definition=subgoal["definition"],
            explanation=subgoal["explanation"],
            dependencies="; ".join(subgoal["dependencies"]),
        )
        messages[-1]["content"] += message + "\n"
    
    for content in contents:
        id = content["id"]
        text = content["text"]
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": f"{id}:\n```{text}```"}],
        })

        # for frame_path in content["frame_paths"]:
        #     frame_base64 = encode_image(frame_path)
        #     messages[-1]["content"].append({
        #         "type": "image_url",
        #         "image_url": {"url": f"data:image/jpeg;base64,{frame_base64}"}
        #     })
    
    common_subgoals = get_response_pydantic(messages, IndexVideoSegmentation)

    ### todo: find the frame where result is depicted for each subgoal
    segment_to_subgoal = {}
    for subgoal in common_subgoals:
        for content_id in subgoal["content_ids"]:
            segment_to_subgoal[content_id] = {
                "title": subgoal["title"],
                "explanation": subgoal["explanation"]
            }
    
    result = []
    for content in contents:
        title = "Custom"
        explanation = "Custom subgoal"
        if content["id"] in segment_to_subgoal:
            title = segment_to_subgoal[content["id"]]["title"]
            explanation = segment_to_subgoal[content["id"]]["explanation"]

        if len(result) > 0 and result[-1]["title"] == title:
            result[-1]["finish"] = content["finish"]
            result[-1]["text"] += content["text"] + f" ({float_to_str(content['finish'])}) "
            result[-1]["frame_paths"].extend(content["frame_paths"])
            result[-1]["contents"].append(content)
        else:
            result.append({
                "start": content["start"],
                "finish": content["finish"],
                "text": f"({float_to_str(content['start'])}) " + content["text"] + f" ({float_to_str(content['finish'])}) ",
                "frame_paths": content["frame_paths"],
                "title": title,
                "explanation": explanation,
                "contents": [content],
            })
    return result

def get_meta_summary_v2(contents, task):
    messages = [
        {"role": "system", "content": "You are a helpful assistant specializing in analyzing and summarizing procedural contents of how-to videos for {task}. Given the contents from a how-to video (i.e., narration and images), extract and summarize the overall context (i.e., goal and outcome) and approach (steps). Your summaries are concise, focused, and avoid unnecessary details, ensuring that the essential information is captured without redundancy.".format(task=task)},
    ]

    messages.append({
        "role": "user",
        "content": [],
    })
    for content in contents:
        text = content["text"]
        messages[-1]["content"].append({
            "type": "text",
            "text": f"{text}"
        })
        # for frame_path in content["frame_paths"]:
        #     frame_base64 = encode_image(frame_path)
        #     messages[-1]["content"].append({
        #         "type": "image_url",
        #         "image_url": {"url": f"data:image/jpeg;base64,{frame_base64}"}
        #     })
    response = get_response_pydantic(messages, MetaSummarySchema)

    return response

def get_subgoal_summaries_v2(subgoals, task):
    messages = [
        {"role": "system", "content": "You are a helpful assistant specializing in analyzing and summarizing procedural information of subgoals in how-to videos for {task}. Given the contents of specific subgoals from a how-to video, extract and summarize the procedural information (context and approach) for each subgoal.".format(task=task)},
    ]

    messages.append({
        "role": "user",
        "content": [],
    })
    for subgoal in subgoals:
        text = subgoal["text"]
        messages[-1]["content"].append({
            "type": "text",
            "text": f"{text}"
        })
        # for frame_path in subgoal["frame_paths"]:
        #     frame_base64 = encode_image(frame_path)
        #     messages[-1]["content"].append({
        #         "type": "image_url",
        #         "image_url": {"url": f"data:image/jpeg;base64,{frame_base64}"}
        #     })
    response = get_response_pydantic(messages, SubgoalSummarySchema)

    return response

def get_subgoal_summary_v2(contents, context, task, subgoal):
    pass
    # messages = [
    #     {"role": "system", "content": "You are a helpful assistant specializing in analyzing and summarizing procedural information of {subgoal} in how-to video for {task}. Given contents of a specific step from a how-to video and the context (i.e., results of previous subgoals as well as the summary of the overall procedure), extract and summarize the unique procedural information according to the provided schema.".format(subgoal=subgoal, task=task)},
    # ]

    # for content in context:
    #     id = content["id"]
    #     text = content["text"]
    #     messages.append({
    #         "role": "user",
    #         "content": [{"type": "text", "text": f"{id}:\n```{text}```"}],
    #     })

    #     for frame_path in content["frame_paths"]:
    #         frame_base64 = encode_image(frame_path)
    #         messages[-1]["content"].append({
    #             "type": "image_url",
    #             "image_url": {"url": f"data:image/jpeg;base64,{frame_base64}"}
    #         })


    # ### TODO: somehow find "good" outcome frames!!!!
    # for content in contents:
    #     id = content["id"]
    #     text = content["text"]
    #     messages.append({
    #         "role": "user",
    #         "content": [{"type": "text", "text": f"{id}:\n```{text}```"}],
    #     })

    #     for frame_path in content["frame_paths"]:
    #         frame_base64 = encode_image(frame_path)
    #         messages[-1]["content"].append({
    #             "type": "image_url",
    #             "image_url": {"url": f"data:image/jpeg;base64,{frame_base64}"}
    #         })

    # response = get_response_pydantic(messages, SubgoalSummarySchema)
    # response["context"] = context

    # return response