from helpers import get_response_pydantic, extend_contents

from pydantic_models.segmentation import StepsSchema, AggStepsSchema, TranscriptAssignmentsSchema, get_segmentation_schema_v4, AggSubgoalsSchema

def assign_transcripts_v4(contents, subgoals, task):
    messages = [
        {"role": "system", "content": "You are a helpful assistant specializing in analyzing tutorial video content. Given a narration of a tutorial video for the task `{task}` and a set of steps, analyze each sentence and find the steps it is talking about. You can specify multiple steps per sentence or leave it empty if it does not belong to any of the steps. Additionally, specify relevance of the sentence to the task at hand.".format(task=task)},
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
        {"role": "system", "content": "You are a helpful assistant specializing in analyzing tutorial video content. Given a narration of a tutorial video for the task `{task}` and a set of steps, segment the entire video based on the steps. Start from the beginning of the video (i.e., 0-th sentence) and sequentially assign matching relevant step label to each subsequent segment of the narration. Make sure that the all the procedurally important parts of the narration are covered.".format(task=task)},
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
    response["segments"] = sorted(response["segments"], key=lambda x: x["start_index"])

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
                new_start = (content["start"] + segments[-1]["finish"]) / 2
                segments[-1]["finish"] = new_start
            else:
                new_start = content["start"]
            segments.append({
                "start": new_start,
                "finish": content["finish"],
                "title": cur_step,
                "text": content["text"],
                "frame_paths": [*content["frame_paths"]],
                "content_ids": [content["id"]],
            })
    return segments

def define_steps_v4(contents, task):
    messages = [
        {"role": "system", "content": "You are a helpful assistant specializing in analyzing tutorial video content. Given a narration of a tutorial video for the task `{task}`, analyze it and generate a comprehensive list of steps presented in the video. Focus on the essence of the steps and avoid including unnecessary details. Ensure that the steps are clear, concise, and cover all the critical procedural information.".format(task=task)},
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

def extract_subgoals_v4(steps, task):
    messages = [
        {"role": "system", "content": "You are a helpful assistant specializing in analyzing tutorial content. You are given a set of generalized steps to perform the task `{task}`. Identify and extract subgoals within this procedure. Each subgoal should represent a distinct, meaningful intermediate stage or outcome within the procedure. Label each subgoal concisely in 1 to 3 words, ensuring each term is both informative and distinct.”".format(task=task)},
        {
            "role": "user",
            "content": "## Generalized Steps:\n" + "\n".join(steps)
        }
    ]
    
    response = get_response_pydantic(messages, AggSubgoalsSchema)
    subgoals = response["subgoals"]
    assignments = response["assignments"]
    for subgoal in subgoals:
        subgoal["original_steps"] = []
        found = 0
        for a in assignments:
            if a["subgoal"] == subgoal["title"]:
                subgoal["original_steps"].append(a["step"])
                found += 1
        if found == 0:
            print("ERROR: Subgoal not found in assignments")
    return subgoals