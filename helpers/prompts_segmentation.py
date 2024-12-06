from typing import Tuple

from helpers import get_response_pydantic, get_response_pydantic_with_message, extend_contents, random_uid

from pydantic_models.segmentation import StepsSchema, AggStepsSchema, TranscriptAssignmentsSchema, get_segmentation_schema_v4, AggSubgoalsSchema

from pydantic_models.segmentation import SubgoalSegmentationSchema, SubgoalSchema, AggregatedSubgoalsSchema, SubgoalsSchema

from pydantic_models.segmentation import SubgoalDescriptionsSchema

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

SYSTEM_PROMPT_DEFINE_STEPS_V4 = """
You are helpful assistant specializing in analyzing how-to videos for procedural tasks.

Extract step-by-step instructions from the tutorial video narration in the choronological order based on the template: `[Action] [Materials] with [Tools] to produce [Outcomes]`

Extraction Rules:

	1.	Identify the `Action` as the main verb describing how materials are being used or transformed.
	2.	Identify the `Materials and Tools` as the list of `objects, entities, or their states` being acted upon or modified by the action. If the materials are not explicitly mentioned, try to imply the materials as much as possible. Make sure that each material is represented separately.
	3.	Identify the `Tools` as the list of `objects, entities, or utilities` enabling the action. If the tools are not explicitly mentioned, try to imply the tools as much as possible. Make sure that each tool is represented separately.
	4.	Identify the `Outcomes` as the list of `objects, entities, or their states` (similar to `Materials`) resulting in applying the action to the materials with the tools. If the outcomes are not explicitly mentioned, try to imply the outcomes as much as possible. Make sure that each outcome is represented separately.

Use this framework to extract clear, actionable steps from the tutorial narration while maintaining logical consistency in the classification of actions, materials, tools, and outcomes.
"""

def __steps_compiler(steps: StepsSchema) -> Tuple[str, bool]:
    compliation_output = ""
    has_error = False
    ### Check if there are dangling objects
    appeared_objects = {}
    is_outcome = {}
    for step in steps:
        index = step["index"]
        for obj in step["inputs"]:
            name = obj["name"]
            last_step = obj["last_step"]
            if name not in appeared_objects:
                appeared_objects[name] = []
            if last_step > 0:
                if last_step not in appeared_objects[name]:
                    compliation_output += f"ERROR: Object {name} did not appear in the step {last_step}, so `last_step` seems incorrect!\n"
            else:
                if len(appeared_objects[name]) > 0 and name in is_outcome:
                    compliation_output += f"ERROR: Object {name} appeared as outcome in one of the step(s) {appeared_objects[name]}, so `last_step` must be fixed!\n"
            appeared_objects[name].append(index)

        for name in step["outcomes"]:
            if name in appeared_objects:
                compliation_output += f"ERROR: Object {name} appeared in steps {appeared_objects[name]}, but it is also outcome of the current step {index}! Since it is a new outcome, it should not have appeared before.\n"
            if name not in appeared_objects:
                appeared_objects[name] = []
            appeared_objects[name].append(index)
            is_outcome[name] = True
    if len(compliation_output) > 0:
        has_error = True
    ### Used only once
    for name, appeared_steps in appeared_objects.items():
        if len(appeared_steps) == 1 and name in is_outcome:
            compliation_output += f"WARNING: Object {name} appeared as outcome in step {appeared_steps[0]}, but was never used! If it is a final outcome of the tutorial skip this. Otherwise, find in which step it should have been used!\n"
    return compliation_output, has_error

def define_steps_v4(contents, task):
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_DEFINE_STEPS_V4.format(task=task)
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"Contents:\n"
            }] + extend_contents(contents),
        },
    ]
    total_tries = 0
    only_warnings = False
    steps = []
    while total_tries < 5:
        total_tries += 1
        response, message = get_response_pydantic_with_message(messages, StepsSchema)
        steps = response["steps"]
        compliation_output, has_error = __steps_compiler(steps)
        if not has_error:
            if only_warnings or len(compliation_output) == 0:
                break
            only_warnings = True
        messages.append({
            "role": "assistant",
            "content": message,
        })
        messages.append({
            "role": "user",
            "content": f"Please correct the following issues and try again:\n{compliation_output}."
        })
    if total_tries == 5:
        print("!!!!!!Run out of tries!")
    print(total_tries, __steps_compiler(steps))
    ## flatten objects
    def __flatten_objects(objs):
        if not objs:
            return []
        return [obj["name"] for obj in objs]
    for step in steps:
        step["inputs"] = __flatten_objects(step["inputs"])
        # step["outcomes"] = __flatten_objects(step["outcomes"])
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

For each subgoal, provide a comprehensive list of descriptions of the expected outcomes of the subgoal.
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

Given a narration of the tutorial and a set of `subgoals` for the task `{task}`, extract the descriptions of the subgoals from the narration. For each subgoal, identify all the relevant segments in the narration, and analyze the contents to extract the description of the subgoal following the template:
`Apply [Method] using [Materials] [Tools] to achieve [Outcomes] to achieve [Subgoal]`

Ensure that descriptions are comprehensive, extracting the right method and all materials, tools, outcomes for each subgoal.
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

    response = get_response_pydantic(messages, SubgoalDescriptionsSchema)
    subgoals = response["subgoals"]
    return subgoals

    # contents_coverage = [""] * len(contents)
    # response["segments"] = sorted(response["segments"], key=lambda x: x["start_index"])

    # mapping = {
    #     "": "",
    # }

    # for segment in response["segments"]:
    #     start = segment["start_index"]
    #     finish = segment["end_index"] + 1
    #     mapping[segment["title"]] = segment["description"]
    #     for i in range(start, finish):
    #         if contents_coverage[i] != "":
    #             prev_title = contents_coverage[i]                
    #             print("Potential ERROR: Overlapping segments", prev_title, segment["title"])
    #         contents_coverage[i] = segment["title"]
    # segments = []
    # for index, step in enumerate(steps):
    #     title = contents_coverage[index]
    #     if len(segments) > 0 and segments[-1]["title"] == title:
    #         ### Extend the current segment
    #         segments[-1]["finish_index"] = index + 1
    #         segments[-1]["text"] += "\n" + step["text"]
    #         segments[-1]["frame_paths"] += [*step["frame_paths"]]
    #     else:
    #         segments.append({
    #             "start_index": index,
    #             "finish_index": index + 1,
    #             "title": title,
    #             "description": mapping[title],
    #             "text": step["text"],
    #             "frame_paths": [*step["frame_paths"]],
    #         })
    # return segments


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