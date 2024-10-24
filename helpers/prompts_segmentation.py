from helpers import get_response_pydantic, extend_contents, extend_subgoals
from helpers import encode_image, get_response_pydantic_with_message

from pydantic_models.segmentation import TaskGraph, get_segmentation_schema, StepsSchema, AggStepsSchema, SubgoalSchema, AllProceduralInformationSchema, TranscriptAssignmentsSchema, get_segmentation_schema_v4


def define_subgoals_v2(contents, task):
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

def align_subgoals_v2(subgoals_per_video, task):
    messages = [
        {"role": "system", "content": "You are a helpful assistant specializing in analyzing and comparing procedural content across different how-to videos about task `{task}`. Given the task graphs (i.e., subgoals and dependencies between them) from multiple how-to videos about the same task, combine them into a single task graph. Where necessary, merge/split subgoals, and generate a unified, comprehensive set of subgoals that is applicable to all the videos. Ensure that the subgoals are based on meaningful intermediate stages of the procedure.".format(task=task)},
    ]
    for video_idx, (_, subgoals) in enumerate(subgoals_per_video.items()):
        messages.append({"role": "user", "content": f"## Video {video_idx + 1}:\n{extend_subgoals(subgoals)}"})
    
    response = get_response_pydantic(messages, TaskGraph)
    subgoals = response["subgoals"]
    return subgoals

def generate_subgoals_v3(contents, subgoals, task):
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

    subgoals = []
    for content in contents:
        if "title" not in content:
            content["title"] = "Custom"
            content["explanation"] = "Custom subgoal"
        if (len(subgoals) > 0):
            if subgoals[-1]["title"] == content["title"] or content["title"] == "Custom":
                subgoals[-1]["finish"] = content["finish"]
                subgoals[-1]["text"] += " " + content["text"]
                subgoals[-1]["frame_paths"] = subgoals[-1]["frame_paths"] + content["frame_paths"]
                subgoals[-1]["content_ids"].append(content["id"])
                continue

        subgoals.append({
            "title": content["title"],
            "explanation": content["explanation"],
            "start": content["start"],
            "finish": content["finish"],
            "text": content["text"],
            "frame_paths": content["frame_paths"],
            "content_ids": [content["id"]]
        })
    return subgoals

def assign_transcripts_v4(contents, subgoals, task):
    messages = [
        {"role": "system", "content": "You are a helpful assistant specializing in analyzing tutorial video content. Given a narration of a tutorial video for a task `{task}` and a set of steps, analyze each sentence and find the steps it is talking about. You can specify multiple steps per sentence or leave it empty if it does not belong to any of the steps. Additionally, specify relevance of the sentence to the task at hand.".format(task=task)},
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"## Steps:\n" + "\n".join(subgoals)
            }]
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"## Contents:\n"
            }] + extend_contents(contents, include_ids=True),
        },
    ]

    
    total_assignments = []
    for i in range(0, len(contents), 20):
        message = {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"Assign steps to the sentences between {i} and {min(i + 19, len(contents) - 1)}:\n"
            }]
        }
        response = get_response_pydantic(messages + [message], TranscriptAssignmentsSchema)
        total_assignments += response["assignments"]

    segments = []
    for index, content in enumerate(contents):
        assignment = None
        for a in total_assignments:
            if a["index"] == index:
                assignment = a
                break

        if assignment is None:
            print("ERROR: Assignment not found for index", index)
            continue
        title = assignment["steps"]
        segments.append({
            "start": content["start"],
            "finish": content["finish"],
            "title": title,
            "text": content["text"],
            "frame_paths": [*content["frame_paths"]],
            "content_ids": [content["id"]],
            "relevance": assignment["relevance"],
        })

    return segments

def segment_video_v4(contents, steps, task):
    messages = [
        {"role": "system", "content": "You are a helpful assistant specializing in analyzing tutorial video content. Given a narration of a tutorial video for a task `{task}` and a set of steps, segment the entire video based on the steps. Start from the beginning of the video (i.e., 0-th sentence) and sequentially assign matching relevant step label to each subsequent segment of the narration. Make sure that the all the procedurally important parts of the narration are covered.".format(task=task)},
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"## Steps:\n" + "\n".join(steps)
            }]
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"## Contents:\n"
            }] + extend_contents(contents, include_ids=True),
        },
    ]

    SegmentationSchema = get_segmentation_schema_v4(None)

    response = get_response_pydantic(messages, SegmentationSchema)

    contents_coverage = [""] * len(contents)
    for segment in response["segments"]:
        start = segment["start_index"]
        finish = segment["end_index"]
        for i in range(start, finish):
            if contents_coverage[i] != "":
                print("Potential ERROR: Overlapping segments", contents_coverage[i], segment["step"])
            contents_coverage[i] = segment["step"]
    segments = []
    for index, content in enumerate(contents):
        cur_step = contents_coverage[index]
        if len(segments) > 0 and cur_step == segments[-1]["title"]:
            ### Extend the current segment
            segments[-1]["finish"] = content["finish"]
            segments[-1]["text"] += " " + content["text"]
            segments[-1]["frame_paths"] = segments[-1]["frame_paths"] + [*content["frame_paths"]]
            segments[-1]["content_ids"].append(content["id"])
        else:
            ### Start a new segment
            if len(segments) > 0:
                segments[-1]["finish"] = content["start"]
            segments.append({
                "start": content["start"],
                "finish": content["finish"],
                "title": cur_step,
                "text": content["text"],
                "frame_paths": [*content["frame_paths"]],
                "content_ids": [content["id"]],
            })
    return segments

def define_steps_v4(contents, task):
    messages = [
        {"role": "system", "content": "You are a helpful assistant specializing in analyzing tutorial video content. Given a narration of a tutorial video for a task `{task}`, analyze it and generate a comprehensive list of steps presented in the video. Focus on the essence of the steps and avoid including unnecessary details. Ensure that the steps are clear, concise, and cover all the critical procedural information.".format(task=task)},
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"Contents:\n"
            }] + extend_contents(contents),
        },
    ]
    
    response = get_response_pydantic(messages, StepsSchema)
    steps = response["steps"]
    return steps

def align_steps_v4(sequence1, sequence2, task):
    sequence1_str = "\n".join(sequence1)
    sequence2_str = "\n".join(sequence2)
    messages = [
        {"role": "system", "content": "You are a helpful assistant specializing in analyzing procedural content across different how-to videos about task `{task}`. Given two lists of steps from two tutorial videos about the task, aggregate them into a single list of steps. Combine similar steps or steps that have the same overall goal. Focus on the essence of the steps and avoid including unnecessary details. Make sure to include all the steps from both videos and specify which aggregated step they belong to.".format(task=task)},
        {"role": "user", "content": f"## Video 1:\n{sequence1_str}"},
        {"role": "user", "content": f"## Video 2:\n{sequence2_str}"}
    ]
    
    response = get_response_pydantic(messages, AggStepsSchema)
    if len(response["assignments_1"]) != len(sequence1) or len(response["assignments_2"]) != len(sequence2):
        print("ERROR: Length of assignments_1 does not match the length of sequence1")

    steps = []
    for agg_step in response["agg_steps"]:
        steps.append({
            "aggregated": agg_step,
            "original_list_1": [],
            "original_list_2": []
        })
        
    for assignments, original_list in [
        ("assignments_1", "original_list_1"),
        ("assignments_2", "original_list_2")
    ]:
        for a in response[assignments]:
            found = 0
            for subgoal in steps:
                if a["agg_step"] == subgoal["aggregated"]:
                    subgoal[original_list].append(a["original_step"])
                    found += 1
            if found == 0:
                print("ERROR: Original step from sequence not found in agg_steps")
            if found > 1:
                print("ERROR: Original step from sequence found in multiple agg_steps")
    return steps

def summarize_steps_v4(steps, task):
    messages = [
        {"role": "system", "content": "You are a helpful assistant specializing in analyzing tutorial content. You are given a set of steps in the task `{task}`. Extract the common goal each of the steps are accomplishing and provide a single COMPREHENSIVE subgoal.".format(task=task)},
        {
            "role": "user",
            "content": "## Steps:\n" + "\n".join(steps)
        }
    ]
    
    response = get_response_pydantic(messages, SubgoalSchema)
    return response

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