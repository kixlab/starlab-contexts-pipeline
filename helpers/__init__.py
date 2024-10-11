import os
import base64
import random
import json

from PIL import Image
from openai import OpenAI

from uuid import uuid4

API_KEY = os.getenv('OPENAI_API_KEY')   
print(API_KEY)
client = OpenAI(
    api_key=API_KEY,
)

SEED = 13774
TEMPERATURE = 0
MODEL_NAME = 'gpt-4o-2024-08-06'


def random_uid():
    return str(uuid4())

def encode_image(image_path):
    with open(image_path, "rb") as image:
        return base64.b64encode(image.read()).decode("utf-8")

def transcribe_audio(audio_path, granularity=["segment"]):
    with open(audio_path, "rb") as audio:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio,
            response_format="verbose_json",
            timestamp_granularities=granularity,
            prompt="Umm, let me think like, hmm... Okay, here's what I'm, like, thinking."
        )
        response = response.to_dict()
        return response
    return None


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
    print("MESSAGES:", json.dumps(messages, indent=2))
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
    
    json_response = response.parsed.dict()

    print("RESPONSE:", json.dumps(json_response, indent=2))
    return json_response

def extend_contents(contents, include_images=False):
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

def extend_subgoals(subgoals):
    message = ""
    for subgoal_idx, subgoal in enumerate(subgoals):
        message += f"- **Subgoal {subgoal_idx + 1}**:\n"
        for k, v in subgoal.items():
            key = k.capitalize()
            value = v if isinstance(v, str) else ", ".join(v)
            message += f"\t- {key}: {value}\n"
    return message



### OTHER CONSTANTS & FUNCTIONS

PATH = "static/results/"
TASK_ID  = "carbonara-short"
META_TITLE = "$meta$"

TASK_DESCRIPTIONS = {
    "carbonara": "How to make Pasta Carbonara?",
    "muffins": "How to bake a muffin?",
    "remove-objects": "How to remove objects from an image in PhotoShop?",
    "binary-search": "How to implement binary search?",
    #"origami-rose": "How to make an origami rose?",
    #"11967": "How to make a paper airplane?",
}

LIBRARY = {
    "carbonara": [
        "https://www.youtube.com/watch?v=75p4UHRIMcU",
        "https://www.youtube.com/watch?v=dzyXBU3dIys",
        "https://www.youtube.com/watch?v=D_2DBLAt57c",
        "https://www.youtube.com/watch?v=3AAdKl1UYZs",
        #"https://www.youtube.com/watch?v=qoHnwOHLiMk",
        #"https://www.youtube.com/watch?v=NqFi90p38N8",
    ],
    "muffins": [
        "https://www.youtube.com/shorts/B-XGIGS4Ipw", # short
        "https://www.youtube.com/shorts/fWp5z_YM07Q", # short
        "https://www.youtube.com/watch?v=aEFvNsBDCWs", # has verbal
        "https://www.youtube.com/watch?v=gN-orgrgvU8", # has verbal
        "https://www.youtube.com/watch?v=cZ2KJPGVwNU", # has verbal
    ],
    "remove-objects": [
        "https://www.youtube.com/watch?v=cTM853qX7cM",
        "https://www.youtube.com/watch?v=WmDG0xp2liA",
        "https://www.youtube.com/watch?v=GC6jRpkwZo0",
        "https://www.youtube.com/watch?v=dSrhB4e5DW0",
        "https://www.youtube.com/watch?v=hshPCtAwSuo",
        "https://www.youtube.com/watch?v=KJWlgG68pUw",
    ],
    "binary-search": [
        "https://www.youtube.com/watch?v=tgVSkMA8joQ",
        "https://www.youtube.com/watch?v=P3YID7liBug",
        "https://www.youtube.com/watch?v=xrMppTpoqdw",
        "https://www.youtube.com/watch?v=Uuyv88Tn9iU",
        "https://www.youtube.com/watch?v=dq9fDmT_CZU",
        "https://www.youtube.com/watch?v=DE-ye0t0oxE",
    ],
    "origami-rose": [
    ],
    "11967": [
        "https://www.youtube.com/watch?v=yJQShkjNn08",
        "https://www.youtube.com/watch?v=yweUoYP1v_o",
        "https://www.youtube.com/watch?v=Ehntsffsx08",
        "https://www.youtube.com/watch?v=tdk9_Xs_CC0",
        "https://www.youtube.com/watch?v=dkhy4vn9HcY",
        "https://www.youtube.com/watch?v=QECo58lV-bE",
        "https://www.youtube.com/watch?v=SMh2sjuEwxM",
        "https://www.youtube.com/watch?v=DaEzhwLFPi8",
        "https://www.youtube.com/watch?v=J_5scvrv0LU",
        "https://www.youtube.com/watch?v=umbBEHlpTfo",
        "https://www.youtube.com/watch?v=pq_INi_4IBI",
        "https://www.youtube.com/watch?v=pYOQutHfCDo",
    ],
}

ALIGNMENT_DEFINITIONS = {
    "`Additional Information`": "what is new",
    "`Alternative Method`": "how is the method different",
    "`Alternative Setting`": "how is the setting different",
    "`Alternative Example`": "how are both method and setting different",
}

VIDEO_SETS = {
    "seen": "previously seen",
    "unseen": "user have not seen",
}

APPROACHES = [
    "approach_1",
    "approach_2",
]

BASELINES = [
    "baseline_1",
    "baseline_2",
]


def str_to_float(str_time):
    return sum(x * float(t) for x, t in zip([3600, 60, 1], str_time.split(":")))

def float_to_str(float_time):
    return str(int(float_time // 3600)) + ":" + str(int((float_time % 3600) // 60)) + ":" + str(int(float_time % 60))

import pysbd

def add_punctuations(text):
    return get_response([
        {"role": "system", "content": "You are a helpful assistant specializing in adding punctuations to how-to video transcriptions. Please add punctuations to the following text. Return a JSON object with the key 'punctuated_text' and the value as the punctuated text."},
        {"role": "user", "content": text},
    ], "json_object")

def segment_into_sentences(text):
    ### do gpt-4 call to get punctuations
    seg = pysbd.Segmenter(language="en", clean=False)
    return seg.segment(text)

def paraphrase(text, former_word, new_word):
    text = text.replace(former_word, new_word)