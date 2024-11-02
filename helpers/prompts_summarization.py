from helpers import get_response_pydantic, extend_contents

from pydantic_models.summarization import StepSummarySchema

def get_step_summary_v4(contents, steps, task):
    if len(steps) == 0:
        return None
    if len(steps) == 1:
        steps = f"step `{steps[0]}`"
    else: 
        steps = f"steps `{'; '.join(steps)}`"

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant specializing in analyzing tutorial video content. Given a narration of a tutorial video for the task `{task}`, identify and extract all procedural information about the {steps}. Make sure to extract the information itself and avoid describing how it was presented.".format(task=task, steps=steps)
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"Contents:\n"
            }] + extend_contents(contents, include_ids=True),
        }
    ]
    response = get_response_pydantic(messages, StepSummarySchema)

    response["frame_paths"] = []
    for content in contents:
        response["frame_paths"] = response["frame_paths"] + content["frame_paths"]        
    return response