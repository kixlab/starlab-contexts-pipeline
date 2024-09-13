import os
import base64
import random
import json

from PIL import Image
from openai import OpenAI

from helpers import META_TITLE, ALIGNMENT_DEFINITIONS

from pydantic_models.subgoal import TaskGraph, VideoSegmentation
from pydantic_models.info_schemas import MetaInformationSchema, SubgoalInformationSchema

from pydantic_models.info_schemas import AlignmentsSchema, AlignmentClassificationSchema, AlignmentHooksSchema

def encode_image(image_path):
    with open(image_path, "rb") as image:
        return base64.b64encode(image.read()).decode("utf-8")

API_KEY = os.getenv('OPENAI_API_KEY')   
print(API_KEY)
client = OpenAI(
    api_key=API_KEY,
)

SEED = 13774
TEMPERATURE = 0.6
MODEL_NAME = 'gpt-4o-2024-08-06'

def get_response(messages, response_format="json_object", retries=1):
    
    generated_text = ""
    finish_reason = ""
    usages = []
    while True:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            seed=SEED,
            temperature=TEMPERATURE,
            response_format={
                "type": response_format,
            },

        )
        generated_text += response.choices[0].message.content
        finish_reason = response.choices[0].finish_reason
        usages.append(response.usage)

        if finish_reason != "length":
            break
        messages.append({"role": "assistant", "content": response.choices[0].message.content})

    # print(f"Finish Reason: {finish_reason}")
    # print(f"Usages: {usages}")
    # print(f"Generated Text: {generated_text}")

    if response_format == "json_object":
        try:
            obj = json.loads(generated_text)
            keys = list(obj.keys())
            if len(keys) == 1:
                return obj[keys[0]]
            else:
                return obj
        except json.JSONDecodeError:
            if retries > 0:
                return get_response(messages, response_format, retries - 1)

    return generated_text


def get_response_pydantic(messages, response_format):
    print(json.dumps(messages, indent=2))

    completion = client.beta.chat.completions.parse(
        model=MODEL_NAME,
        messages=messages,
        seed=SEED,
        temperature=TEMPERATURE,
        response_format=response_format,
    )

    response = completion.choices[0].message
    if (response.refusal):
        print("REFUSED: ", response.refusal)
        return None
    
    json_response = response.parsed.get_dict()

    print(json.dumps(json_response, indent=2))
    return json_response



def generate_custom_steps(subtitles):
    messages = [
        {"role": "system", "content": "You are a helpful assistant specializing in analyzing and defining comprehensive subgoals generalizable across different how-to videos about the same task."},
        {"role": "user", "content": "Summarizes the subtitles of a video into important steps in the procedural task (steps should be based on meaningful intermediate stages of the process). You must use the subtitles to generate the steps. Return a JSON list with the following structure: [{'start': float, 'finish': float, 'title': string, 'text': string}]"},
        {"role": "user", "content": f"Subtitles:\n```{json.dumps(subtitles)}```"}
    ]
    return get_response(messages)

def define_common_steps(videos):
    if (len(videos) == 0):
        return []
    
    if (len(videos) == 1):
        messages = [
            {"role": "system", "content": "You are a helpful assistant specializing in analyzing and refining subgoals that are generalizable across different how-to videos for the same task. Your goal is to ensure that the subgoals are broad enough to encompass diverse content yet specific enough to capture all critical procedural steps."},
            {"role": "user", "content": "Based on the the how-to video, identify and define subgoals that are generalizable across different videos covering the same task. Ensure that the subgoals encompass all the procedural information in the video."},
            {"role": "user", "content": f"Narration:\n```{json.dumps(videos[0])}```"}
        ]
    else: 
        ## intial subgoals
        messages = [
            {"role": "system", "content": "You are a helpful assistant specializing in analyzing and refining subgoals that are generalizable across different how-to videos for the same task. Your goal is to ensure that the subgoals are broad enough to encompass diverse content yet specific enough to capture all critical procedural steps."},
            {"role": "user", "content": "Based on two how-to videos about the same task, identify and define subgoals that are generalizable for both videos. Ensure that the subgoals encompass all the procedural information in the video."},
            {"role": "user", "content": "\n".join((f"Narration {idx + 1}:\n```{json.dumps(video)}```") for idx, video in enumerate(videos[:2]))}
        ]
    
    subgoals = get_response_pydantic(messages, TaskGraph)

    ## refine subgoals
    if len(videos) < 3:
        return subgoals
    
    for video in videos[2:]:
        messages = [
            {"role": "system", "content": "You are a helpful assistant specializing in analyzing and refining subgoals that are generalizable across different how-to videos for the same task. Your goal is to ensure that the subgoals are broad enough to encompass diverse content yet specific enough to capture all critical procedural steps."},
            {"role": "user", "content": "Given the the how-to video and the initial definitions of subgoals, refine the set of subgoals (i.e., add/remove/change subgoal definitions). Ensure that new subgoals are `equivalent` to previous set, but at the same time, comprehensively cover all procedural content in the current video. Make sure the subgoals are at the right level of abstraction: specific enough to classify diverse content effectively, but not so broad that they lose their utility."},
            {"role": "user", "content": f"Narration:\n```{json.dumps(video)}```"},
            {"role": "user", "content": f"Subgoals:\n```{json.dumps(subgoals)}```"}
        ]
        subgoals = get_response_pydantic(messages, TaskGraph)
    
    return subgoals

def generate_common_steps(subtitles, subgoals):
    messages = [
        {"role": "system", "content": "You are a helpful assistant specializing in analyzing and segmenting content in how-to videos according to given subgoals."},
        {"role": "user", "content": "Given the subtitles of the how-to video and subgoals, segment the video according to subgoals. If some subtitles do not belong to any subgoal, define a custom subgoals & label them with `(custom)` tag. Make sure to preserve the order of the subtitles in the video."},
        {"role": "user", "content": f"Subtitles:\n```{json.dumps(subtitles)}```"},
        {"role": "user", "content": f"Subgoals:\n```{json.dumps(subgoals)}```"}
    ]

    common_steps = get_response_pydantic(messages, VideoSegmentation)
    return common_steps

def get_meta_alignments(summary1, summary2):
    messages = [
        {
            "role": "system",
            "content": "You are a detailed-oriented assistant specializing in analyzing and comparing procedural content across different how-to videos."
        },
        {
            "role": "user",
            "content": "Given the procedural information from two videos about the same task, identify all the new content in the new video compared to the previous video. Folow the schema provided below to structure your response."
        },
        {"role": "user", "content": f"New Video:\n```{json.dumps(summary1)}```"},
        {"role": "user", "content": f"Previous Video:\n```{json.dumps(summary2)}```"},
    ]
    response = get_response_pydantic(messages, AlignmentsSchema)
    return response

def get_subgoal_alignments(title, context1, context2, summary1, summary2):
    messages = [
        {
            "role": "system",
            "content": "You are a detailed-oriented assistant specializing in analyzing and comparing procedural content across different how-to videos."
        },
        {
            "role": "user",
            "content": "Given the procedural information from two videos about the same task, identify all the new content in the new video compared to the previous video. Folow the schema provided below to structure your response."
        },
        {"role": "user", "content": f"New Video:\n```{json.dumps(summary1)}```"},
        {"role": "user", "content": f"Previous Video:\n```{json.dumps(summary2)}```"},
    ]
    response = get_response_pydantic(messages, AlignmentsSchema)
    return response

def get_meta_summary(title, source):
    messages = [
        {"role": "system", "content": "You are a helpful assistant specializing in analyzing and summarizing procedural information from how-to videos. Your summaries are concise, focused, and avoid unnecessary details, ensuring that the essential information is captured without redundancy."},
        {"role": "user", "content": f"Given the narration of the how-to video, extract and summarize the information according to the provided schema."},
        {"role": "user", "content": f"Narration:\n```{source}```"},
    ]

    response = get_response_pydantic(messages, MetaInformationSchema)

    return {
        "title": title,
        **response,
    }

def get_subgoal_summary(title, source, context, process):
    messages = [
        {"role": "system", "content": "You are a helpful assistant specializing in analyzing and summarizing procedural information from how-to videos. Your summaries are concise, focused, and ensure that essential information is captured without redundancy."},
        {"role": "user", "content": f"Given the narration of a specific step from a how-to video and the context (i.e., results of previous subgoals as well as the summary of the overall procedure), extract and summarize the unique procedural information according to the provided schema."},
        {"role": "user","content": f"Context:\n```{json.dumps(context)}\n{json.dumps(process)}```"},
        {"role": "user","content": f"Narration:\n```{json.dumps(source)}```"},
    ]

    response = get_response_pydantic(messages, SubgoalInformationSchema)

    return {
        "title": title,
        **response,
    }

def get_alignment_classification(alignment, title, new_meta, prev_meta, new_subgoal, prev_subgoal):
    messages = [
        {"role": "system", "content": "You are a helpful assistant specializing in analyzing and classifying the information in terms of comparison and alignment between different how-to videos."},
    ]
    
    new_video = None
    prev_video = None

    if title == META_TITLE:
        new_video = new_meta
        prev_video = prev_meta
    else:
        new_video = {
            "overall_context": new_meta,
            f"{title}": new_subgoal,
        }
        prev_video = {
            "overall_context": prev_meta,
            f"{title}": prev_subgoal,
        }
    messages.extend([
        {"role": "user", "content": "Given the new information from a new video compared to the previous video as well as the original context from both videos, analyze and classify the new information into following categories:\n- **Additional Information**: The new information is considered as `additional information` if the methods and the setting are fundamentally the same (i.e., same instructions, rationale, or tips, subgoal, context, materials, outcome) , but the new video provides additional information that is not present in the previous video.\n- **Alternative Method**: The new information is considered as `alternative method` if the methods are different (i.e., different instructions, rationale, or tips), but the setting are fundamentally the same (i.e., same subgoal, context, materials, outcome).\n- **Alternative Setting**: The new information is considered as `alternative setting` if the settings are different (i.e., different subgoal, context, materials, outcome), but the methods are fundamentally the same.\n- **Alternative Example**: The new information is considered as `alternative example` if the new video provides a different settings and methods that are not present in the previous video (i.e., different instructions, rationale, or tips, subgoal, context, materials, outcome).\n."},
        {"role": "user", "content": f"Information:\n```{json.dumps(alignment)}```"},
        {"role": "user", "content": f"New Video:\n```{json.dumps(new_video)}```"},
        {"role": "user", "content": f"Previous Video:\n```{json.dumps(prev_video)}```"},
    ])

    response = get_response_pydantic(messages, AlignmentClassificationSchema)
    return response

def get_hooks_0(classification, alignments):
    messages = [
        {"role": "system", "content": "You are a helpful assistant specializing in analyzing and identifying connections between different pieces of information and generating metacognitive prompts to help people learn about procedural knowledge."},
        {"role": "user", "content": f"Given the information sourced from different how-to videos, cluster them based on their helpfulness to the learners as {classification} and organize them under metacognitive prompts specific to {classification}."},
        {"role": "user", "content": f"```{json.dumps(alignments)}```"},
    ]

    response = get_response_pydantic(messages, AlignmentHooksSchema)
    return response

def get_hooks(video_set, classification, title, alignments):

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant specializing in analyzing and comparing content across different how-to videos in terms of {title}.".format(title=title)
        },
        {
            "role": "user",
            "content": "You are given a list of information alignments between a new video and multiple {video_set} videos for a subgoal {subgoal}. The alignments are due to the new video containing {alignment_type}. Your task is to combine the information from previous videos that pertain to the same content in the new video. For each such content from the new video, list all the {video_set} videos that it relates to and describe the alignment. Lastly, for each combination of information form {video_set} videos and the new video content, determine a hook (a 3 word sentence that highlight {alignment_type_definition}) that can be used to generally describe the alignment between the two and can attract the user's attention.".format(
                video_set=video_set,
                subgoal=title,
                alignment_type=classification,
                alignment_type_definition=ALIGNMENT_DEFINITIONS[classification]
            )
        },
        {
            "role": "user",
            "content": f"Here is the list of information alignments between the new video and multiple previous videos:\n```\n{json.dumps(alignments)}\n```"
        }
    ]

    response = get_response_pydantic(messages, response_format=AlignmentHooksSchema)
    return response["hooks"]