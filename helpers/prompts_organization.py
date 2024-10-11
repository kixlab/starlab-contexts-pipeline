from helpers import get_response_pydantic, extend_contents

from pydantic_models.organization import SummarizedAlignmentSchema, NotableInformationSchema, HookSchema

def get_alignments_summary_v2(contents, task):
    messages = [
        {
            "role": "system",
            "content": [{
                "type": "text",
                "text": "You are a helpful assistant specializing in analyzing and summarizing procedural contents of how-to videos for task `{task}`. You are given the descriptions of `notable procedural information` in the current video, their reasoning and comparison to several previous videos. Aggregate the information into a single description, reasoning, and comparison.".format(task=task)
            }],
        },
        {
            "role": "user",
            "content": extend_contents(contents),
        }
    ]

    response = get_response_pydantic(messages, SummarizedAlignmentSchema)
    return response

def get_notable_v2(contents, task):
    messages = [
        {
            "role": "system",
            "content": [{
                "type": "text",
                "text": "You are a helpful assistant specializing in analyzing and summarizing procedural contents of how-to videos for task `{task}`. You are given the descriptions of `notable procedural information` in the current video compared to various previous videos. Create a 5-word (or fewer) title summarizing the information, followed by a detailed description of the notable information. Focus only on the information itself without referencing or comparing it to other videos.".format(task=task)
            }],
        },
        {
            "role": "user",
            "content": extend_contents(contents),
        }
    ]

    response = get_response_pydantic(messages, NotableInformationSchema)
    return response

def get_hook_v2(contents, task):
    messages = [
        {
            "role": "system",
            "content": [{
                "type": "text",
                "text": "You are a helpful assistant specializing in analyzing and summarizing procedural contents of how-to videos for task `{task}`. You are given the descriptions of `notable/interesting procedural information` in other videos. Generate a conversational, but informative summary that captures the essence of the information and sparks interest in the new content or overall comparison and provide a concise description or elaboration that explains the hook in more detail.".format(task=task)
            }],
        },
        {
            "role": "user",
            "content": extend_contents(contents),
        }
    ]

    response = get_response_pydantic(messages, HookSchema)
    return response