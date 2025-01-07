from helpers import get_response_pydantic, get_response_pydantic_with_message, extend_contents, random_uid

from pydantic_models.instructions import TimedInstructionsSchema

SYS_PROMPT_INSTRUCTIONS = """
Extract instructions from tutorial video subtitles for the task: {task}.
The extracted instructions should adhere to the following requirements:
1. Focus on a single action at a time.
2. Useful for completing the task.
3. Clear and concise.
4. Self-sufficient and do not refer to earlier or later instructions.
5. Free of any ambiguous terms or phrases.
6. Written in the imperative mood.
"""
# 7. Structured to include materials, tools, actions, and outcomes where applicable.


def get_instructions(contents, task):
    messages = [
        {   
            "role": "system",
            "content": SYS_PROMPT_INSTRUCTIONS.format(task=task),
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"## Narration:\n"
            }] + extend_contents(contents, include_images=False, include_ids=True),
        },
    ]

    response = get_response_pydantic(messages, TimedInstructionsSchema)
    instructions = response["instructions"]
    return instructions