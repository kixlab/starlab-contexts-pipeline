import os
import base64
import json

from openai import OpenAI
from uuid import uuid4

API_KEY = os.getenv('OPENAI_API_KEY')   
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

def get_response_pydantic_with_message(messages, response_format):
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
        return None, response.choices[0].message.content
    
    json_response = response.parsed.dict()

    print("RESPONSE:", json.dumps(json_response, indent=2))
    return json_response, completion.choices[0].message.content

def extend_contents(contents, include_images=False, include_ids=False):
    extended_contents = []
    for index, content in enumerate(contents):
        text = content["text"]
        if include_ids:
            text = f"{index}. {text}"
        extended_contents.append({
            "type": "text",
            "text": text,
        })
        if include_images:
            for frame_path in content["frame_paths"]:
                frame_base64 = encode_image(frame_path)
                extended_contents.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{frame_base64}"}
                })
    return extended_contents

APPROACHES = [
    "approach_1",
]

BASELINES = [
    "baseline_1",
]

def str_to_float(str_time):
    return sum(x * float(t) for x, t in zip([3600, 60, 1], str_time.split(":")))

def float_to_str(float_time):
    return str(int(float_time // 3600)) + ":" + str(int((float_time % 3600) // 60)) + ":" + str(int(float_time % 60))

import pysbd

def segment_into_sentences(text):
    seg = pysbd.Segmenter(language="en", clean=False)
    return seg.segment(text)