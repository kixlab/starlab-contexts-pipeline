from helpers import get_response_pydantic, extend_contents

from pydantic_models.organization import SummarizedAlignmentSchema2, GroupsSchema, GroupSchema

def get_notable_v4(contents, subgoal, aspect, task):
    messages = [
        {
            "role": "system",
            "content": [{
                "type": "text",
                "text": "You are a helpful assistant specializing in analyzing and summarizing procedural contents from tutorials for the task `{task}`. You are given pieces of `notable or interesting procedural contents` about {aspect} in the step {subgoal} sourced from a single tutorial. It also contains comparison of each content with other tutorials about the same task. Provide a concise summary of the contents and its descriptions, reasonings (i.e., why is it in the video?), and comparisons.".format(task=task, aspect=aspect, subgoal=subgoal)
            }],
        },
        {
            "role": "user",
            "content": extend_contents(contents),
        }
    ]

    response = get_response_pydantic(messages, SummarizedAlignmentSchema2)
    return response

def get_hook_v4(contents, subgoal, relation, aspect, task):
    messages = [
        {
            "role": "system",
            "content": [{
                "type": "text",
                "text": "You are a helpful assistant specializing in analyzing and summarizing procedural contents from tutorials for the task `{task}`. You are given pieces of {relation} `notable or interesting procedural contents` about {aspect} in the step {subgoal} sourced from other tutorials. It also contains comparison of each content to the current tutorial. Analyze the given contents and create a title that captures the essence of the information and sparks interest in the new content or overall comparison. Then, provide a concise summary of the contents and its descriptions and comparisons.".format(task=task, relation=relation, aspect=aspect, subgoal=subgoal)
            }],
        },
        {
            "role": "user",
            "content": extend_contents(contents, include_ids=True),
        }
    ]

    response = get_response_pydantic(messages, GroupSchema)
    return response

def get_hooks_v4(contents, subgoal, relation, aspect, task):
    messages = [
        {
            "role": "system",
            "content": [{
                "type": "text",
                "text": "You are a helpful assistant specializing in analyzing and summarizing procedural contents from tutorials for the task `{task}`. You are given pieces of {relation} `notable or interesting procedural contents` about {aspect} in the step {subgoal} sourced from other tutorials. It also contains comparison of each content to the current tutorial. Identify and combine similar pieces of contents and create a short and informative titles for each group of contents. Then, provide short summaries for each group in terms of descriptions and comparisons.".format(task=task, relation=relation, aspect=aspect, subgoal=subgoal)
            }],
        },
        {
            "role": "user",
            "content": extend_contents(contents, include_ids=True),
        }
    ]

    response = get_response_pydantic(messages, GroupsSchema)

    groups = response["groups"]
    for group in groups:
        group["links"] = []

    coverage = [0 for _ in range(len(contents))]
    for mapping in response["assignments"]:
        index = mapping["content_index"]
        coverage[index] = 1
        for group in groups:
            if group["title"] == mapping["group"]:
                group["links"].append(index)
                break
    for i in range(len(coverage)):
        if coverage[i] == 0:
            print(f"WARNING: Content {i} is not assigned to any group.")
    return groups