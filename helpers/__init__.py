import tiktoken
import os
import base64
import json
import time
import threading
import asyncio
import numpy as np
import random
from dataclasses import dataclass, field

from openai import OpenAI
from uuid import uuid4

from anthropic import AnthropicBedrock

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from stop_words import get_stop_words

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')   

SEED = 13774
TEMPERATURE = 0
MAX_TOKENS = 32700 ### response should be max 32k tokens
MODEL_NAME_OPENAI = 'gpt-4.1-mini-2025-04-14'

REASONING_EFFORT = "minimal" ### "minimal", "low", "medium", "high"

client_openai = OpenAI(
    api_key=OPENAI_API_KEY,
)

def random_uid():
    return str(uuid4())

def get_response_pydantic_openai(messages, response_format, model=None):
    if model is None:
        model = MODEL_NAME_OPENAI

    json_obj = response_format
    if hasattr(response_format, "model_json_schema"):
        json_obj = {
            "schema": response_format.model_json_schema(),
            "name": response_format.__name__,
        }
    
    response_format_dict = {
        "type": "json_schema",
        "json_schema": {
            **json_obj,
            "strict": True,
        },
    }

    basic_args = {
        "model": model,
        "seed": SEED,
        "messages": messages,
        "max_completion_tokens": MAX_TOKENS,
        "response_format": response_format_dict,
    }
    variable_args = {}
    if "gpt-5" in model:
        variable_args["reasoning_effort"] = REASONING_EFFORT
    else:
        variable_args["temperature"] = TEMPERATURE
    
    # print("requesting openai...")
    # print(json.dumps(messages, indent=4))

    response_raw = client_openai.chat.completions.with_raw_response.create(
        **basic_args,
        **variable_args,
    )
    
    completion = response_raw.parse()
    response = completion.choices[0].message.content

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        raise Exception(f"refusal: {response}")

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

def split_system_prompt(messages):
    system_prompt = "You are a helpful assistant."
    for message in messages:
        if message["role"] == "system":
            system_prompt = message["content"]
            break

    messages = [message for message in messages if message["role"] != "system"]

    return messages, system_prompt

FORMATTING_PROMPT = """
Please format the final output in the JSON format specified within the <format> tag. Enclose the JSON in <response> tags (i.e., <response>{{JSON-parsable-text}}</response>).
<format>
{format}
</format>
"""

def get_response_pydantic_anthropic(messages, response_format, model=None):
    if model is None:
        model = MODEL_NAME_ANTHROPIC
    
    if hasattr(response_format, "model_json_schema"):
        format_schema = json.dumps(response_format.model_json_schema())
    else:
        format_schema = json.dumps(response_format)
    messages += [
        {
            "role": "user",
            "content": FORMATTING_PROMPT.format(format=format_schema)
        }
    ]
    messages, system_prompt = split_system_prompt(messages)
    completion = client_anthropic.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=messages,
        temperature=TEMPERATURE,
    )
    response = completion.content[0].text
    try:
        json_text = response.split("<response>")[1].split("</response>")[0]
        return json.loads(json_text)
    except json.JSONDecodeError:
        print(f"refusal: {response}")
        raise Exception(f"refusal: {response}")

# ============== Embedding Utilities ==============

PER_TEXT_TOKEN_LIMIT = 2048
PER_ARRAY_TOKEN_LIMIT = 300000
TOTAL_ARRAY_LENGTH = 2048

EMBEDDING_MODEL_OPENAI = "text-embedding-3-large"

en_stop_words = get_stop_words('en')
### fine-tune the model
bert_model = SentenceTransformer("all-MiniLM-L6-v2")
# model.fit(
#     train_objectives=[(train_dataloader, train_loss)],
#     epochs=1,
#     warmup_steps=100,
#     optimizer_params={'lr': 1e-4},
# )

def count_tokens(text, model="gpt-4"):
    encoding = tiktoken.encoding_for_model(model)
    n_tokens = len(encoding.encode(text))
    return n_tokens

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

def get_openai_embedding(texts, model=None):
    if model is None:
        model = EMBEDDING_MODEL_OPENAI
    chunks = [[]]
    total_length = 0
    for text in texts:
        if text == "":
            text = " "
        text = text.replace("\n", " ")
        cur_tokens = count_tokens(text)
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

def perform_embedding(embedding_method, texts):
    """
    Embed the texts using the appropriate embedding method.
    """
    if embedding_method == "tfidf":
        return tfidf_embedding(texts)
    elif embedding_method == "bert":
        return bert_embedding(texts)
    elif embedding_method == "openai":
        return get_openai_embedding(texts) ### TODO: reimplement to use batching if over the limit, but for now just direct call
    else:
        raise ValueError(f"Invalid embedding method: {embedding_method}")

def str_to_float(str_time):
    return sum(x * float(t) for x, t in zip([3600, 60, 1], str_time.split(":")))

def float_to_str(float_time):
    return str(int(float_time // 3600)) + ":" + str(int((float_time % 3600) // 60)) + ":" + str(int(float_time % 60))

import pysbd

def segment_into_sentences(text):
    seg = pysbd.Segmenter(language="en", clean=False)
    return seg.segment(text)

# ============== Batch/Parallel LLM utilities (JSONL) ==============
MAX_REQUESTS_PER_MINUTE = 30000 * 0.5 ## 30k requests per minute
MAX_TOKENS_PER_MINUTE = 150000000 * 0.5 ## 150M tokens per minute
MAX_ATTEMPTS = 6

BATCH_PATH = "./static/results/lm-batches/"
DEFAULT_CALL_FUNCTION = get_response_pydantic_openai
# DEFAULT_CALL_FUNCTION = get_response_pydantic_anthropic

### TODO: deprecate this
def get_response_pydantic(messages, response_format, model=None):
    return DEFAULT_CALL_FUNCTION(messages, response_format, model=model)

def _ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

def llm_get_requests_path(batch_id):
    return os.path.join(BATCH_PATH, f"{batch_id}_llm_requests.jsonl")

def llm_get_responses_path(batch_id):
    return os.path.join(BATCH_PATH, f"{batch_id}_llm_responses.jsonl")

_append_lock = threading.Lock()
def _append_jsonl(path: str, obj) -> None:
    line = json.dumps(obj, ensure_ascii=False)
    with _append_lock:
        _ensure_parent_dir(path)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

def llm_enqueue_request(batch_id, messages, response_format, model, options=None, request_id=None):
    req_id = request_id or random_uid()
    response_format_dict = {
        "schema": response_format.model_json_schema(),
        "name": response_format.__name__,
    }
    record = {
        "request_id": req_id,
        "model": model,
        "messages": messages,
        "response_format": response_format_dict,
        "options": options or {},
    }
    path = llm_get_requests_path(batch_id)
    _append_jsonl(path, record)
    return req_id

def llm_get_response(batch_id, request_id):
    path = llm_get_responses_path(batch_id)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            request_json = record[0]
            result = record[1]
            if request_json["request_id"] == request_id:
                if "errors" in result:
                    raise ValueError(f"Errors: {result['errors']}")
                return result
    raise ValueError(f"Request ID not found: {request_id}")

def llm_get_responses(batch_id):
    path = llm_get_responses_path(batch_id)
    responses = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            request_json = record[0]
            result = record[1]
            if "errors" in result:
                raise ValueError(f"Errors: {result['errors']}")
            responses[request_json["request_id"]] = result
    return responses

# ---------- simple token bucket ----------
class TokenBucket:
    def __init__(self, max_per_minute: float):
        self.capacity = float(max_per_minute)
        self.tokens = float(max_per_minute)
        self.rate_per_sec = float(max_per_minute) / 60.0
        self.last = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self, amount: float = 1.0):
        while True:
            async with self._lock:
                now = time.time()
                elapsed = now - self.last
                if elapsed > 0:
                    self.tokens = min(self.capacity, self.tokens + elapsed * self.rate_per_sec)
                    self.last = now
                if self.tokens >= amount:
                    self.tokens -= amount
                    return
                needed = amount - self.tokens
                wait_secs = needed / self.rate_per_sec if self.rate_per_sec > 0 else 0.05
            await asyncio.sleep(wait_secs)

# ---------- status tracking ----------
@dataclass
class StatusTracker:
    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0
    num_other_errors: int = 0
    time_of_last_rate_limit_error: float = 0.0

# ---------- request envelope ----------
@dataclass
class APIRequest:
    task_id: int
    request_json: dict
    attempts_left: int
    # We’ll treat MAX_TOKENS (response cap) as token consumption. If you want to be tighter,
    # add prompt token counting and sum(prompt_tokens + response_cap).
    token_consumption: int = field(default_factory=lambda: MAX_TOKENS)

# ---------- error helpers ----------
def _is_rate_limit_error(err: Exception) -> bool:
    msg = str(err).lower()
    return "rate limit" in msg or "429" in msg or "too many requests" in msg or "throttl" in msg

def _is_retryable_http_error(err: Exception) -> bool:
    # crude but effective; catches 5xx/network resets/timeouts from most SDKs
    # include OpenAI's refusal error  
    msg = str(err).lower()
    return any(k in msg for k in ["timeout", "temporar", "reset", "connection", "server error", "503", "502", "504", "refusal"])

def _classify_error(err: Exception) -> str:
    if _is_rate_limit_error(err):
        return "rate"
    if _is_retryable_http_error(err):
        return "retryable"
    return "fatal"

# ---------- backoff with jitter ----------
def _backoff_seconds(attempt_idx: int, base: float = 1.0, cap: float = 60.0) -> float:
    # attempt_idx starts at 1 for first retry
    expo = base * (2 ** (attempt_idx - 1))
    with_jitter = expo * (0.5 + random.random())  # 0.5x–1.5x jitter
    return min(with_jitter, cap)

# ---------- main: process requests ----------
async def process_api_requests(batch_id, max_concurrency=64, seconds_to_pause_after_limit_error=60.0):
    requests_path = llm_get_requests_path(batch_id)
    responses_path = llm_get_responses_path(batch_id)

    # buckets for throttling
    req_bucket = TokenBucket(MAX_REQUESTS_PER_MINUTE)
    tok_bucket = TokenBucket(MAX_TOKENS_PER_MINUTE)

    status = StatusTracker()
    queue = asyncio.Queue(maxsize=max_concurrency * 4)

    # producer: stream JSONL -> queue
    async def producer():
        nonlocal status
        task_id = 0
        with open(requests_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                req = APIRequest(
                    task_id=task_id,
                    request_json=record,
                    attempts_left=MAX_ATTEMPTS,
                    token_consumption=estimate_token_consumption(record)
                )
                task_id += 1
                status.num_tasks_started += 1
                status.num_tasks_in_progress += 1
                await queue.put(req)
        # tell consumers we're done
        for _ in range(max_concurrency):
            await queue.put(None)  # type: ignore

    # consumer: take from queue, throttle, call SDK, save result
    async def worker(worker_id: int):
        nonlocal status
        while True:
            req = await queue.get()
            if req is None:
                return
            await req_bucket.acquire(1.0)
            await tok_bucket.acquire(float(req.token_consumption))
            attempt = 0
            errors = []
            while True:
                try:
                    def _call():
                        r = req.request_json
                        model = r.get("model")
                        messages = r.get("messages")
                        response_format = r.get("response_format")
                        options = r.get("options") or {}
                        return DEFAULT_CALL_FUNCTION(messages, response_format, model=model)
                    result = await asyncio.to_thread(_call)
                    _append_jsonl(responses_path, [req.request_json, result])
                    status.num_tasks_succeeded += 1
                    status.num_tasks_in_progress -= 1
                    break
                except Exception as e:
                    errors.append(str(e))
                    kind = _classify_error(e)
                    if kind == "rate":
                        status.num_rate_limit_errors += 1
                        status.time_of_last_rate_limit_error = time.time()
                        await asyncio.sleep(seconds_to_pause_after_limit_error)
                    elif kind == "retryable":
                        status.num_api_errors += 1
                    else:
                        status.num_other_errors += 1
                        _append_jsonl(responses_path, [req.request_json, {"errors": errors}])
                        status.num_tasks_failed += 1
                        status.num_tasks_in_progress -= 1
                        break
                    # retry gate
                    req.attempts_left -= 1
                    attempt += 1
                    if req.attempts_left <= 0:
                        _append_jsonl(responses_path, [req.request_json, {"errors": errors}])
                        status.num_tasks_failed += 1
                        status.num_tasks_in_progress -= 1
                        break

                    # backoff before retry; also lightly sleep to avoid hot-looping the queue
                    await asyncio.sleep(_backoff_seconds(attempt))
    await asyncio.gather(producer(), *[worker(i) for i in range(max_concurrency)])
    print(f"Done. Success={status.num_tasks_succeeded}  "
                 f"Failed={status.num_tasks_failed}  "
                 f"RateLimitErrs={status.num_rate_limit_errors}")

MESSAGE_OVERHEAD_TOKENS = 4
REPLY_PRIMING_TOKENS = 2

def _encoding_for(model: str | None):
    try:
        if model:
            return tiktoken.encoding_for_model(model)
    except Exception:
        pass
    return tiktoken.get_encoding("cl100k_base")

def _count_text_tokens(text: str, enc) -> int:
    if not text:
        return 0
    return len(enc.encode(text))

def _extract_text_from_content(content):
    """
    content can be:
      - str
      - list[ {type: "text"|"image_url"|... , ...} ]
    We only count "text".
    """
    if isinstance(content, str):
        return [content]
    texts = []
    if isinstance(content, list):
        for part in content:
            try:
                if isinstance(part, dict) and part.get("type") == "text":
                    texts.append(part.get("text", "") or "")
            except Exception:
                continue
    return texts

def estimate_prompt_tokens_from_messages(messages, model: str | None) -> int:
    """
    Rough chat token estimator:
      - +4 per message (role/name/content wrappers)
      - + tokens(content)
      - if 'name' present, subtract 1 (role omitted)
      - +2 once for assistant reply priming
    """
    enc = _encoding_for(model)
    total = 0
    for msg in messages or []:
        total += MESSAGE_OVERHEAD_TOKENS
        name = msg.get("name")
        if name:
            total -= 1  # role omission rule of thumb

        # handle content
        contents = _extract_text_from_content(msg.get("content"))
        for t in contents:
            total += _count_text_tokens(t, enc)

        # in case some messages still use "text" alongside "content"
        if "text" in msg and isinstance(msg["text"], str):
            total += _count_text_tokens(msg["text"], enc)

    total += REPLY_PRIMING_TOKENS
    return total

def estimate_token_consumption(request_json: dict) -> int:
    model = request_json.get("model")
    messages = request_json.get("messages", [])
    prompt_tokens = estimate_prompt_tokens_from_messages(messages, model)

    # Optional: include your Anthropic formatting prompt if that path will add it.
    if model and model.startswith("us.anthropic."):
        prompt_tokens += estimate_prompt_tokens_from_messages(
            [{"role": "user", "content": FORMATTING_PROMPT}], model
        )

    print(f"Prompt tokens: {prompt_tokens}")

    return prompt_tokens