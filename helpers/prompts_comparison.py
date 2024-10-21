from pydantic_models.comparison import AlignmentsSchema3, AlignmentsSchema2, AlignmentsSchema

from helpers import get_response_pydantic, get_response_pydantic_with_message, extend_contents


### TODO: May need to add images and try again!!
def get_subgoal_alignments_v2(contents1, contents2, subgoal, task):
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
                "text": f"## Previous Video:`\n"
            }] + extend_contents(contents2, False),
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"## Current Video:`\n"
            }] + extend_contents(contents1, False),
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
                "text": "You are a helpful assistant specializing in analyzing and comparing procedural content across different how-to videos about task `{task}`. Given contents from the current and previous videos, analyze and compare the information from each video and provide a comprehensive list of new supplementary and contradictory contents presented in the current video. For each piece of content, focus on one specific point at a time, avoid combining multiple details.".format(task=task)
            }],
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"## Previous Video:`\n"
            }] + extend_contents(contents2, False),
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"## Current Video:`\n"
            }] + extend_contents(contents1, False),
        }
    ]

    response = get_response_pydantic(messages, AlignmentsSchema)
    return response["alignments"]


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
                "text": f"## Previous Video:`\n"
            }] + extend_contents(contents2, False),
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"## Current Video:`\n"
            }] + extend_contents(contents1, False),
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
                "text": f"## Previous Video:`\n"
            }] + extend_contents(contents2, False),
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"## Current Video:`\n"
            }] + extend_contents(contents1, False),
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

def get_transcript_alignments_v3(contents1, contents2, task):
    messages = [
        {
            "role": "system",
            "content": [{
                "type": "text",
                "text": "You are a helpful assistant specializing in analyzing and comparing procedural content about task `{task}`. Given contents from the two videos, analyze and compare the information from each video and provide a comprehensive list of new supplementary and contradictory contents presented in the current video. For each piece of content, focus on one specific point at a time, avoid combining multiple details.".format(task=task)
            }],
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"## Previous Video:`\n"
            }] + extend_contents(contents2, False),
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"## Current Video:`\n"
            }] + extend_contents(contents1, False),
        },
    ]

    alignments = []

    tries = 5
    while tries > 0:
        tries -= 1
        response, message = get_response_pydantic_with_message(messages, AlignmentsSchema2)

        messages.append({
            "role": "assistant",
            "content": message
        })
        messages.append({
            "role": "user",
            "content": [{
                "type": "text",
                "text": "If applicable, provide more supplementary and contradictory contents presented in the current video compared to the previous video."
            }]
        })

        found_any = False
        
        for alignment in response["supplementary_information"]:
            alignment["classification"] = "supplementary_" + alignment["classification"]
            alignments.append(alignment)
            found_any = True
        for alignment in response["contradictory_information"]:
            alignment["classification"] = "contradictory_" + alignment["classification"]
            alignments.append(alignment)
            found_any = True
        if found_any is False:
            break

    return alignments


def get_meta_alignments_v4(contents1, contents2, task):
    messages = [
        {
            "role": "system",
            "content": [{
                "type": "text",
                "text": "You are a helpful assistant specializing in analyzing and comparing procedural content across different how-to videos about task `{task}`. Given contents from the current and previous videos, analyze and compare the information from each video and provide a comprehensive list of supplementary and contradictory contents present only in the current video with respect to previous video. For each piece of content, focus on one specific point (i.e., detail, aspect) at a time, avoid combining multiple details together.".format(task=task)
            }],
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"## Previous Video:`\n"
            }] + extend_contents(contents2, False),
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"## Current Video:`\n"
            }] + extend_contents(contents1, False),
        }
    ]

    response = get_response_pydantic(messages, AlignmentsSchema3)
    alignments = []
    for alignment in response["supplementary_information"]:
        alignment["classification"] = "supplementary_" + alignment["classification"]
        alignments.append(alignment)
    for alignment in response["contradictory_information"]:
        alignment["classification"] = "contradictory_" + alignment["classification"]
        alignments.append(alignment)

    return alignments