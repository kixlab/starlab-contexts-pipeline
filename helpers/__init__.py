import tiktoken
import os
import base64
import json

from openai import OpenAI
from uuid import uuid4
import numpy as np

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')   

client_openai = OpenAI(
    api_key=OPENAI_API_KEY,
)

EMBEDDING_MODEL_OPENAI = "text-embedding-3-large"

SEED = 13774
TEMPERATURE = 0
MAX_TOKENS = 4096
# MODEL_NAME_OPENAI = 'gpt-5-mini-2025-08-07' #reasoning
# MODEL_NAME_OPENAI = 'gpt-4.1-2025-04-14'
MODEL_NAME_OPENAI = 'gpt-4.1-mini-2025-04-14'
# MODEL_NAME_OPENAI = 'gpt-4.1-nano-2025-04-14'
# MODEL_NAME_OPENAI = 'gpt-4o-mini-2024-07-18'

REASONING_EFFORT = "low" ### "low", "medium", "high"

PER_TEXT_TOKEN_LIMIT = 2048
PER_ARRAY_TOKEN_LIMIT = 300000
TOTAL_ARRAY_LENGTH = 2048

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from stop_words import get_stop_words


### fine-tune the model
bert_model = SentenceTransformer("all-MiniLM-L6-v2")
# model.fit(
#     train_objectives=[(train_dataloader, train_loss)],
#     epochs=1,
#     warmup_steps=100,
#     optimizer_params={'lr': 1e-4},
# )

en_stop_words = get_stop_words('en')

def random_uid():
    return str(uuid4())

def count_tokens(text, model="gpt-4"):
    encoding = tiktoken.encoding_for_model(model)
    n_tokens = len(encoding.encode(text))
    return n_tokens

def encode_image(image_path):
    with open(image_path, "rb") as image:
        return base64.b64encode(image.read()).decode("utf-8")

def messages_to_str(messages):
    return "\n".join([f"ROLE: {message['role']}\n{message['content']}" for message in messages])

def transcribe_audio(audio_path, granularity=["segment"]):
    with open(audio_path, "rb") as audio:
        response = client_openai.audio.transcriptions.create(
            model="whisper-1",
            file=audio,
            response_format="verbose_json",
            timestamp_granularities=granularity,
            prompt="Umm, let me think like, hmm... Okay, here's what I'm, like, thinking."
        )
        response = response.to_dict()
        return response

def get_openai_embedding(texts, model=EMBEDDING_MODEL_OPENAI):
    chunks = [[]]
    total_length = 0
    for text in texts:
        if text == "":
            text = " "
        text = text.replace("\n", " ")
        cur_tokens = count_tokens(text, model=model)
        if cur_tokens > PER_TEXT_TOKEN_LIMIT:
            raise ValueError(f"Text is too long: {text}")
        if total_length + cur_tokens > PER_ARRAY_TOKEN_LIMIT or len(chunks[-1]) + 1 > TOTAL_ARRAY_LENGTH:
            chunks.append([])
            total_length = 0
        chunks[-1].append(text)
        total_length += cur_tokens
    result = []
    for chunk in chunks:
        if len(chunk) == 0:
            continue
        response = client_openai.embeddings.create(
            input=chunk,
            model=model,
        )
        result.extend([data.embedding for data in response.data])
    return np.array(result)

def bert_embedding(texts):
    if len(texts) == 0:
        return np.array([])

    for i in range(len(texts)):
        if texts[i] == "":
            texts[i] = " "
    embeddings = bert_model.encode(texts)
    return embeddings

def tfidf_embedding(texts):
    if len(texts) == 0:
        return np.array([])
    vectorizer = TfidfVectorizer(
        stop_words=en_stop_words,
        max_features=100,
        max_df=0.9,
        # min_df=0.2,
        smooth_idf=True,
        norm='l2',
        ngram_range=(1, 2),
    )
    embeddings = vectorizer.fit_transform(texts)
    return np.array(embeddings.toarray())


def get_response_pydantic_openai(messages, response_format, model=None):
    if model is None:
        model = MODEL_NAME_OPENAI
    print("MODEL: ", model)
    print("MESSAGES:", messages_to_str(messages))

    if 'gpt-5' in model:
        completion = client_openai.chat.completions.parse(
            model=model,
            messages=messages,
            seed=SEED,
            response_format=response_format,
            reasoning_effort=REASONING_EFFORT,
        )
    else:
        completion = client_openai.chat.completions.parse(
            model=model,
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

def get_response(messages, response_format="json_object", retries=1, model=None):
    if model is None:
        model = MODEL_NAME_OPENAI
    generated_text = ""
    finish_reason = ""
    usages = []
    while True:
        response = client_openai.chat.completions.create(
            model=model,
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
    

# def get_response_pydantic_with_message(messages, response_format):
#     print("MESSAGES:", json.dumps(messages, indent=2))
#     completion = client_openai.beta.chat.completions.parse(
#         model=MODEL_NAME_OPENAI,
#         messages=messages,
#         seed=SEED,
#         temperature=TEMPERATURE,
#         response_format=response_format,
#     )

#     response = completion.choices[0].message
#     if (response.refusal):
#         print("REFUSED: ", response.refusal)
#         return None, response.choices[0].message.content
    
#     json_response = response.parsed.dict()

#     print("RESPONSE:", json.dumps(json_response, indent=2))
#     return json_response, completion.choices[0].message.content

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

def perform_embedding(embedding_method, texts):
    """
    Embed the texts using the appropriate embedding method.
    """
    if embedding_method == "tfidf":
        return tfidf_embedding(texts)
    elif embedding_method == "bert":
        return bert_embedding(texts)
    elif embedding_method == "openai":
        return get_openai_embedding(texts)
    else:
        raise ValueError(f"Invalid embedding method: {embedding_method}")


def split_system_prompt(messages):
    system_prompt = "You are a helpful assistant."
    for message in messages:
        if message["role"] == "system":
            system_prompt = message["content"]
            break

    messages = [message for message in messages if message["role"] != "system"]

    return messages, system_prompt

from anthropic import AnthropicBedrock, AsyncAnthropicBedrock

ANTHROPIC_ACCESS_KEY = os.getenv('ANTHROPIC_ACCESS_KEY')
ANTHROPIC_SECRET_KEY = os.getenv('ANTHROPIC_SECRET_KEY')

ANTHROPIC_REGION = 'us-west-2'
MODEL_NAME_ANTHROPIC = "us.anthropic.claude-3-5-haiku-20241022-v1:0" ### fastest 0.8$/4$
# MODEL_NAME_ANTHROPIC = "us.anthropic.claude-3-7-sonnet-20250219-v1:0" ### slower, ### 3$/15$

client_anthropic = AnthropicBedrock(
    aws_access_key=ANTHROPIC_ACCESS_KEY,
    aws_secret_key=ANTHROPIC_SECRET_KEY,
    aws_region=ANTHROPIC_REGION,
)

def get_response_anthropic(messages, model=None):
    if model is None:
        model = MODEL_NAME_ANTHROPIC
    messages, system_prompt = split_system_prompt(messages)
    response = client_anthropic.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=messages,
        temperature=TEMPERATURE,
    )
    return response.content[0].text


client_anthropic_async = AsyncAnthropicBedrock(
    aws_access_key=ANTHROPIC_ACCESS_KEY,
    aws_secret_key=ANTHROPIC_SECRET_KEY,
    aws_region=ANTHROPIC_REGION,
)

async def get_response_anthropic_async(messages, model=None):
    if model is None:
        model = MODEL_NAME_ANTHROPIC
    messages, system_prompt = split_system_prompt(messages)
    response = await client_anthropic_async.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=messages,
        temperature=TEMPERATURE,
    )
    return response.content[0].text


FORMATTING_PROMPT = """
Please format the final output in the JSON format specified within the <format> tag. Enclose the JSON in <response> tags (i.e., <response>{{JSON-parsable-text}}</response>).
<format>
{format}
</format>
"""


def get_response_pydantic_anthropic(messages, response_format, model=None):
    format_schema = response_format.model_json_schema()
    messages += [
        {
            "role": "user",
            "content": FORMATTING_PROMPT.format(format=format_schema)
        }
    ]
    print("MODEL: ", model)
    print("MESSAGES:", messages_to_str(messages))
    
    response = get_response_anthropic(messages, model)
    print("RAW RESPONSE:", response)
    try:
        json_text = response.split("<response>")[1].split("</response>")[0]
        json_response = json.loads(json_text)
    except json.JSONDecodeError:
        json_response = None
        print("ERROR PARSING JSON")

    return json_response

def get_response_anthropic_with_tries(messages, response_format, tries=3):
    for _ in range(tries):
        try:
            response = get_response_pydantic_anthropic(messages, response_format)
            if response:
                return response
            else:
                print("ERROR PARSING JSON")
                continue
        except Exception as e:
            print(f"Error: {e}")
            continue
    return None


# RESPONSE_FUNC = get_response_pydantic_anthropic
# RESPONSE_FUNC = get_response_anthropic_with_tries
RESPONSE_FUNC = get_response_pydantic_openai

def get_response_pydantic(messages, response_format, model=None):

    json_response = RESPONSE_FUNC(messages, response_format, model)
    
    return json_response