from pydantic_models.comparison import AlignmentsSchema2

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

### TODO: May need to add images and try again!!
def get_subgoal_alignments_v3(contents1, contents2, subgoal, task):
    messages = [
        {
            "role": "system",
            "content": [{
                "type": "text",
                "text": "You are a helpful assistant specializing in analyzing and comparing procedural content across different how-to videos about task `{task}`. Given contents from the current and previous videos for the subgoal `{subgoal}`, analyze and compare the information from each video and provide a comprehensive list of new supplementary and contradictory subgoal contents presented in the current video. For each piece of content, focus on one specific point at a time, avoid combining multiple details.".format(task=task, subgoal=subgoal)
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

    response = get_response_pydantic(messages, AlignmentsSchema2)
    alignments = []
    for alignment in response["supplementary_information"]:
        alignment["classification"] = "supplementary_" + alignment["classification"]
        alignments.append(alignment)
    for alignment in response["contradictory_information"]:
        alignment["classification"] = "contradictory_" + alignment["classification"]
        alignments.append(alignment)

    return alignments

def get_meta_alignments_v3(contents1, contents2, task):
    messages = [
        {
            "role": "system",
            "content": [{
                "type": "text",
                "text": "You are a helpful assistant specializing in analyzing and comparing procedural content across different how-to videos about task `{task}`. Given contents from the current and previous videos, analyze and compare the information from each video and provide a comprehensive list of new supplementary and contradictory contents presented in the current video. For each piece of content, focus on one specific point at a time, avoid combining multiple details.".format(task=task)
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

    response = get_response_pydantic(messages, AlignmentsSchema2)
    alignments = []
    for alignment in response["supplementary_information"]:
        alignment["classification"] = "supplementary_" + alignment["classification"]
        alignments.append(alignment)
    for alignment in response["contradictory_information"]:
        alignment["classification"] = "contradictory_" + alignment["classification"]
        alignments.append(alignment)

    return alignments