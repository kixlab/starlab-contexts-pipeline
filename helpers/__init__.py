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


RESPONSE_FUNC = get_response_pydantic_openai

def get_response_pydantic(messages, response_format, model=None):

    json_response = RESPONSE_FUNC(messages, response_format, model)
    
    return json_response