from helpers import get_response_pydantic, extend_contents, random_uid

from pydantic_models.segmentation import StepsSchema, AggStepsSchema, TranscriptAssignmentsSchema, get_segmentation_schema_v4, AggSubgoalsSchema

from pydantic_models.segmentation import SubgoalSegmentationSchema, SubgoalSchema, AggregatedSubgoalsSchema, SubgoalsSchema

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
        {"role": "system", "content": "You are a helpful assistant specializing in analyzing tutorial video content. Given a narration of a tutorial video for the task `{task}` and a set of steps, segment the entire video based on the steps. Start from the beginning of the video (i.e., 0-th sentence) and sequentially assign matching relevant step label to each subsequent segment of the narration. Make sure that all the procedurally important parts of the narration are covered.".format(task=task)},
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
        finish = segment["end_index"] + 1
        step = segment["step"]
        for i in range(start, finish):
            if contents_coverage[i] != "":
                print("Potential ERROR: Overlapping segments", contents_coverage[i], step)
            contents_coverage[i] = step
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
    sequence1_str = "\n".join([f"{i}. {step}" for i, step in enumerate(sequence1)])
    sequence2_str = "\n".join([f"{i}. {step}" for i, step in enumerate(sequence2)])
    messages = [
        {"role": "system", "content": "You are a helpful assistant specializing in analyzing procedural content across different how-to videos about task `{task}`. Given two lists of steps from two tutorial videos about the task, aggregate them into a single list of steps. Combine similar steps or steps that have the same overall goal. Focus on the essence of the steps and avoid including unnecessary details. Make sure to include all the steps from both videos and specify which aggregated step they belong to.".format(task=task)},
        {"role": "user", "content": f"## Video 1:\n{sequence1_str}"},
        {"role": "user", "content": f"## Video 2:\n{sequence2_str}"}
    ]
    
    response = get_response_pydantic(messages, AggStepsSchema)
    steps = response["agg_steps"]
    coverage_1 = [0] * len(sequence1)
    coverage_2 = [0] * len(sequence2)
    for step in steps:
        original_steps_1 = []
        original_steps_2 = []
        for a in step['original_steps_1']:
            original_steps_1.append(sequence1[a])
            if coverage_1[a] == 1:
                print("ERROR: Original step from sequence 1 already covered:", sequence1[a])
            coverage_1[a] = 1
        for a in step['original_steps_2']:
            original_steps_2.append(sequence2[a])
            if coverage_2[a] == 1:
                print("ERROR: Original step from sequence 2 already covered:", sequence2[a])
            coverage_2[a] = 1
        step['original_steps_1'] = original_steps_1
        step['original_steps_2'] = original_steps_2
    if sum(coverage_1) != len(sequence1) or sum(coverage_2) != len(sequence2):
        print("ERROR: Not all steps covered")


    # if len(response["assignments_1"]) != len(sequence1) or len(response["assignments_2"]) != len(sequence2):
    #     print("ERROR: Length of assignments_1 does not match the length of sequence1")

    # steps = []
    # for agg_step in response["agg_steps"]:
    #     steps.append({
    #         "agg_step": agg_step,
    #         "original_steps_1": [],
    #         "original_steps_2": []
    #     })
        
    # for assignments, original_steps in [
    #     ("assignments_1", "original_steps_1"),
    #     ("assignments_2", "original_steps_2")
    # ]:
    #     for a in response[assignments]:
    #         found = 0
    #         for subgoal in steps:
    #             if a["agg_step"] == subgoal["agg_step"]:
    #                 subgoal[original_steps].append(a["original_step"])
    #                 found += 1
    #         if found == 0:
    #             print("ERROR: Original step from sequence not found in agg_steps")
    #         if found > 1:
    #             print("ERROR: Original step from sequence found in multiple agg_steps")
    return steps

def extract_subgoals_v4(steps, task):
    steps_str = "\n".join([f"{i}. {step}" for i, step in enumerate(steps)])
    messages = [
        {"role": "system", "content": "You are a helpful assistant specializing in analyzing tutorial content. You are given a set of generalized steps to perform the task `{task}`. Identify and extract subgoals within this procedure. Each subgoal should represent a distinct, meaningful intermediate stage or outcome within the procedure. Label each subgoal concisely in 1 to 3 words, ensuring each term is both informative and distinct.".format(task=task)},
        {
            "role": "user",
            "content": "## Generalized Steps:\n" + steps_str
        }
    ]
    
    response = get_response_pydantic(messages, AggSubgoalsSchema)
    subgoals = response["subgoals"]
    coverage = [0] * len(steps)
    for subgoal in subgoals:
        original_steps = []
        for a in subgoal["original_steps"]:
            original_steps.append(steps[a])
            if coverage[a] == 1:
                print("ERROR: Original step already covered:", steps[a])
            coverage[a] = 1
        subgoal["original_steps"] = original_steps
    if sum(coverage) != len(steps):
        print("ERROR: Not all steps covered")
    # assignments = response["assignments"]
    # for subgoal in subgoals:
    #     subgoal["original_steps"] = []
    #     found = 0
    #     for a in assignments:
    #         if a["subgoal"] == subgoal["title"]:
    #             subgoal["original_steps"].append(a["step"])
    #             found += 1
    #     if found == 0:
    #         print("ERROR: Subgoal not found in assignments")
    return subgoals



## V5

SYSTEM_PROMPT_DEFINE_INITIAL_SUBGOALS_V5 = """
You specialize in creating novice-friendly tutorials for procedural tasks.

Define a set of `subgoals` that can guide novices through the task `{task}`. Ensure each `subgoal` meets the following requirements:
    a) Goal-Oriented: Represents an overarching objective or subtask that structures the process into meaningful phases.
    b) Procedural: Describes a procedural objective that is essential for progressing in the task.
    c) Generalizable: Accommodates variations in materials, tools, methods, and outcomes.

Output the `subgoals` for the task `{task}`.
"""

def define_initial_subgoals_v5(task):
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_DEFINE_INITIAL_SUBGOALS_V5.format(task=task),
        },
    ]
    
    response = get_response_pydantic(messages, SubgoalsSchema)
    subgoals = response["subgoals"]
    return subgoals

SYSTEM_PROMPT_EXTRACT_SUBGOAL_SEGMENTS_V5 = """
You specialize in analyzing tutorials for procedural tasks.

Given a narration of a tutorial video and a set of `subgoals` for the task `{task}`, segment the narration based on the given `subgoals`.

Step 1: Analyze the narration and identify meaningful segments. There are 3 types of segments:
    a) A segment that directly or roughly corresponds to one of the provided `subgoals`.
    b) A segment that contains procedural information, but does not correspond to any of the provided `subgoals`.
    c) A segment that does not contain procedural information. Usually, at the beginning or end of the video.

Step 2-a: If a segment is of type (a), assign the corresponding `subgoal` to it and, if necessary, update the description of the `subgoal` so that it also covers the content of the segment.
Step 2-b: If a segment is of type (b), define a new `subgoal` that follows these requirements:
    a) Goal-Oriented: Represents an overarching objective or subtask that structures the process into meaningful phases.
    b) Procedural: Describes a procedural objective that is essential for progressing in the task.
    c) Generalizable: Accommodates variations in materials, tools, methods, and outcomes.
Step 2-c: If a segment is of type (c), ignore it.

Output the segmented narration based on the `subgoals`.
"""


def extract_subgoal_segments_v5(contents, subgoals, task):
    messages = [
        {   "role": "system",
            "content": SYSTEM_PROMPT_EXTRACT_SUBGOAL_SEGMENTS_V5.format(task=task),
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"## Subgoals:\n"
            }] + extend_contents(subgoals, include_ids=False),
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"## Narration:\n"
            }] + extend_contents(contents, include_ids=True),
        },
    ]

    response = get_response_pydantic(messages, SubgoalSegmentationSchema)

    contents_coverage = [""] * len(contents)
    response["segments"] = sorted(response["segments"], key=lambda x: x["start_index"])

    mapping = {
        "": {
            "title": "",
            "description": ""
        }
    }

    for segment in response["segments"]:
        start = segment["start_index"]
        finish = segment["end_index"] + 1
        id = f"segment-{random_uid()}"
        mapping[id] = {
            "title": segment["title"],
            "description": segment["description"],
        }
        for i in range(start, finish):
            if contents_coverage[i] != "":
                prev_id = contents_coverage[i]                
                print("Potential ERROR: Overlapping segments", mapping[prev_id], mapping[id])
            contents_coverage[i] = id
    segments = []
    for index, content in enumerate(contents):
        id = contents_coverage[index]
        if len(segments) > 0 and segments[-1]["id"] == id:
            ### Extend the current segment
            segments[-1]["finish"] = content["finish"]
            segments[-1]["text"] += " " + content["text"]
            segments[-1]["frame_paths"] += [*content["frame_paths"]]
            segments[-1]["content_ids"].append(content["id"])
        else:
            ### Start a new segment
            if len(segments) > 0:
                new_start = (content["start"] + segments[-1]["finish"]) / 2
                segments[-1]["finish"] = new_start
            else:
                new_start = content["start"]
            segments.append({
                "id": id,
                "start": new_start,
                "finish": content["finish"],
                "title": mapping[id]["title"],
                "description": mapping[id]["description"],
                "text": content["text"],
                "frame_paths": [*content["frame_paths"]],
                "content_ids": [content["id"]],
            })
    return segments


def aggregate_subgoals_v5(subgoals, task):
    subgoals_str = "\n".join(subgoals)
    messages = [
        {"role": "system", "content": "You are a helpful assistant specializing in analyzing tutorial content. You are given a set of subgoals about the task `{task}`. Analyze and aggregate these subgoals into a new subgoal that can cover all the information that can be attributed to provided subgoals.".format(task=task)},
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"## Subgoals:\n{subgoals_str}"
            }]
        },
    ]
    
    response = get_response_pydantic(messages, SubgoalSchema)
    return response


SYSTEM_PROMPT_AGGREGATE_SUBGOAL_SET_V5 = """
You specialize in analyzing tutorials for procedural tasks.

Given `subgoals` extracted from different tutorials for the task `{task}`, generate a single comprehensive set of `parent subgoals` that generalize all the given subgoals. Follow these steps:

Step 1: Analyze and group the given `subgoals` based on similar objectives or their relevance to the same overarching subtask.
    - Each group must represent a distinct, meaningful objective or subtask within the task.
    - Each subgoal should belong to only one group.
    - Create a new group if a subgoal does not align with existing groups.

Step 2: Synthesize a single `parent subgoal` for each group to represent the shared objective or subtask of its subgoals.
    - Ensure the parent subgoal is concise, goal-oriented, and abstract, avoiding specific tools, materials, methods, or outcomes.
    - Incorporate the descriptions of all subgoals in the group while maintaining generalizability to variations across tutorials.
    - Ensure the parent subgoals are distinct and non-overlapping to structure the task into a comprehensive set of phases.

Output the final set of `parent subgoals` with their corresponding grouped subgoals."
"""

def aggregate_subgoal_set_v5(contents, task):
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_AGGREGATE_SUBGOAL_SET_V5.format(task=task)},
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"## Subgoals:\n"
            }] + extend_contents(contents, include_ids=True),
        },
    ]

    print(AggregatedSubgoalsSchema.model_json_schema())

    response = get_response_pydantic(messages, AggregatedSubgoalsSchema)

    subgoals = []
    covered = [0] * len(contents)
    for subgoal in response["subgoals"]:
        new_subgoal = {
            "subgoal_id": f"subgoal-{random_uid()}",
            "title": subgoal["title"],
            "description": subgoal["description"],
            "original_subgoals": []
        }
        if len(subgoal["original_subgoal_ids"]) == 0:
            print("ERROR: Empty original subgoal ids", subgoal)
            continue

        for i in subgoal["original_subgoal_ids"]:
            if covered[i] == 1:
                print("ERROR: Original subgoal already covered:", contents[i])
                continue
            covered[i] = 1
            for original_subgoal in contents[i]["original_subgoals"]:
                new_subgoal["original_subgoals"].append({
                    "id": original_subgoal["id"],
                    "title": original_subgoal["title"],
                    "description": original_subgoal["description"]
                })
        subgoals.append(new_subgoal)

    for i in range(len(contents)):
        if covered[i] == 0:
            print("ERROR: Original subgoal not covered:", contents[i])
            subgoals.append({
                "subgoal_id": f"subgoal-{random_uid()}",
                "title": contents[i]["title"],
                "description": contents[i]["description"],
                "original_subgoals": [*contents[i]["original_subgoals"]]
            })
    return subgoals