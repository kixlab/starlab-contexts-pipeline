"""
Vision-Language Model (VLM) client with parallel batch processing support.

Supports:
- OpenAI API (GPT-4, GPT-5 series)
- AWS Bedrock (Anthropic Claude models)
- Images via base64 encoding
- Parallel request processing with rate limiting
- Automatic retry with exponential backoff
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import random
import re
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import tiktoken
from openai import BadRequestError, OpenAI
from anthropic import AnthropicBedrockMantle
from google import genai
from google.genai import types as genai_types
from uuid import uuid4

from .config import ModelConfig

logger = logging.getLogger(__name__)


class LLMParseError(Exception):
    """Raised when an LLM response cannot be parsed. Treated as retryable."""


# ============== Configuration ==============

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_ACCESS_KEY = os.getenv("ANTHROPIC_ACCESS_KEY")
ANTHROPIC_SECRET_KEY = os.getenv("ANTHROPIC_SECRET_KEY")
ANTHROPIC_REGION = os.getenv("ANTHROPIC_REGION", "us-west-2")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Default models
DEFAULT_MODEL_OPENAI = "gpt-4.1-mini-2025-04-14"
DEFAULT_MODEL_ANTHROPIC = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
DEFAULT_MODEL_GEMINI = "gemini-2.5-flash"

# Generation parameters
SEED = 13774
TEMPERATURE = 0.0
REASONING_EFFORT = "medium"  # For GPT-5 models: "minimal", "low", "medium", "high"

# Model-specific max completion tokens
MODEL_MAX_TOKENS = {
    # OpenAI models
    "gpt-4": 8192,
    "gpt-4-turbo": 4096,
    "gpt-4o": 16384,
    "gpt-4o-mini": 16384,
    "gpt-4.1-mini": 16384,
    "gpt-5": 32768,
    # Anthropic models (via Bedrock)
    "claude-opus-4-7": 128000,
    "claude-sonnet-4-6": 64000,
    "claude-3-5-sonnet": 8192,
    "claude-3-5-haiku": 8192,
    "claude-3-opus": 4096,
    "claude-3-sonnet": 4096,
    "claude-3-haiku": 4096,
    # Gemini models
    "gemini-3.1-pro": 65536,
    "gemini-3.1-flash-lite": 65536,
    "gemini-3-flash": 65536,
    "gemini-2.5-pro": 8192,
    "gemini-2.5-flash": 8192,
}


def get_max_tokens_for_model(model: str) -> int:
    """Get the maximum completion tokens for a given model."""
    # Check for exact match first
    if model in MODEL_MAX_TOKENS:
        return MODEL_MAX_TOKENS[model]

    # Check for partial matches, sorted by length (longest first)
    # This ensures "gpt-4o-mini" matches before "gpt-4o" before "gpt-4"
    sorted_keys = sorted(MODEL_MAX_TOKENS.keys(), key=len, reverse=True)
    for key in sorted_keys:
        if key in model:
            return MODEL_MAX_TOKENS[key]

    # Default to conservative value for unknown models
    return 4096

# Rate limits (conservative defaults)
MAX_REQUESTS_PER_MINUTE = 15000  # 50% of OpenAI's 30k limit
MAX_TOKENS_PER_MINUTE = 75_000_000  # 50% of OpenAI's 150M limit
MAX_ATTEMPTS = 6

# Batch processing
BATCH_PATH = "./static/lm-batches/"

# Token estimation
MESSAGE_OVERHEAD_TOKENS = 4
REPLY_PRIMING_TOKENS = 2


# ============== Client Setup ==============

_openai_client: Optional[OpenAI] = None
_anthropic_client: Optional[AnthropicBedrockMantle] = None
_gemini_client: Optional[genai.Client] = None


def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


def _get_anthropic_client() -> AnthropicBedrockMantle:
    global _anthropic_client
    if _anthropic_client is None:
        if not ANTHROPIC_ACCESS_KEY or not ANTHROPIC_SECRET_KEY:
            raise ValueError(
                "ANTHROPIC_ACCESS_KEY and ANTHROPIC_SECRET_KEY environment variables must be set"
            )
        _anthropic_client = AnthropicBedrockMantle(
            aws_access_key=ANTHROPIC_ACCESS_KEY,
            aws_secret_key=ANTHROPIC_SECRET_KEY,
            aws_region=ANTHROPIC_REGION,
        )
    return _anthropic_client


def _get_gemini_client() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    return _gemini_client


# ============== Image Utilities ==============


def encode_image(image_path: str | Path) -> str:
    """Encode an image file to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


@dataclass
class MessageContent:
    """
    Flexible message content that supports true interleaving of text and images.

    Content is specified as a list of blocks, where each block is a dict with uniform structure:
    - Text: {"type": "text", "text": "..."}
    - Image from path: {"type": "image_path", "path": "...", "detail": "auto"}

    This format is JSON-serializable and suitable for logging.

    Usage:
        # Text only
        MessageContent(content=[
            {"type": "text", "text": "Describe this video"}
        ])

        # Text with images (non-interleaved)
        MessageContent(content=[
            {"type": "text", "text": "What's happening in these frames?"},
            {"type": "image_path", "path": "frame1.jpg"},
            {"type": "image_path", "path": "frame2.jpg"}
        ])

        # True interleaving
        MessageContent(content=[
            {"type": "text", "text": "At 0-5s, the person says: 'First, take the flour'"},
            {"type": "image_path", "path": "frame_002.jpg"},
            {"type": "text", "text": "At 5-10s, they say: 'Mix it with water'"},
            {"type": "image_path", "path": "frame_007.jpg"},
        ])
    """

    content: List[Dict[str, Any]]

    def to_dict(self) -> List[Dict[str, Any]]:
        """Convert to a JSON-serializable dictionary."""
        return { "content": self.content }

    @classmethod
    def from_dict(cls, data: List[Dict[str, Any]]) -> MessageContent:
        """Create MessageContent from a list of blocks."""
        return cls(content=data["content"])

    def to_openai_content(self) -> List[Dict[str, Any]]:
        """Convert to OpenAI API content format."""
        result: List[Dict[str, Any]] = []

        for block in self.content:
            block_type = block.get("type")

            if block_type == "text":
                result.append({"type": "text", "text": block["text"]})

            elif block_type == "image_path":
                encoded = encode_image(block["path"])
                detail = block.get("detail", "auto")
                result.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded}",
                            "detail": detail,
                        },
                    }
                )

        return result

    def to_anthropic_content(self) -> List[Dict[str, Any]]:
        """Convert to Anthropic API content format."""
        result: List[Dict[str, Any]] = []

        for block in self.content:
            block_type = block.get("type")

            if block_type == "text":
                result.append({"type": "text", "text": block["text"]})

            elif block_type == "image_path":
                encoded = encode_image(block["path"])
                # Detect media type from extension
                suffix = Path(block["path"]).suffix.lower()
                media_type = {
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".png": "image/png",
                    ".gif": "image/gif",
                    ".webp": "image/webp",
                }.get(suffix, "image/jpeg")

                result.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": encoded,
                        },
                    }
                )

        return result

    def to_gemini_content(self) -> List[genai_types.Part]:
        """Convert to Gemini API content format (list of Parts)."""
        result: List[genai_types.Part] = []

        for block in self.content:
            block_type = block.get("type")

            if block_type == "text":
                result.append(genai_types.Part.from_text(text=block["text"]))

            elif block_type == "image_path":
                suffix = Path(block["path"]).suffix.lower()
                mime_type = {
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".png": "image/png",
                    ".gif": "image/gif",
                    ".webp": "image/webp",
                }.get(suffix, "image/jpeg")

                with open(block["path"], "rb") as f:
                    raw_bytes = f.read()

                result.append(genai_types.Part.from_bytes(data=raw_bytes, mime_type=mime_type))

        return result

    @classmethod
    def from_template(cls, template: str, **kwargs) -> MessageContent:
        """Build MessageContent from a template with mixed str and MessageContent values.

        String values are substituted via str.format(). MessageContent values are
        spliced in-place, so their blocks (text, images, ...) appear exactly where
        the placeholder was in the template.

        Usage:
            MessageContent.from_template(
                "Here is the video: {video}\\nNow answer: {question}",
                video=video_mc,        # MessageContent — spliced in place
                question="What tool?", # str — normal substitution
            )
        """
        # Separate str vs MessageContent kwargs
        str_kwargs: Dict[str, str] = {}
        mc_kwargs: Dict[str, MessageContent] = {}
        for k, v in kwargs.items():
            if isinstance(v, MessageContent):
                mc_kwargs[k] = v
            else:
                str_kwargs[k] = str(v)

        if not mc_kwargs:
            return cls(content=[{"type": "text", "text": template.format(**str_kwargs)}])

        # Replace MC placeholders with null-byte markers before str.format()
        markers: Dict[str, str] = {}
        for key in mc_kwargs:
            marker = f"\x00MC_{key}\x00"
            markers[key] = marker
            template = template.replace(f"{{{key}}}", marker)

        # Format the remaining string placeholders
        text = template.format(**str_kwargs)

        # Split on markers (preserving them) and interleave
        pattern = "(" + "|".join(re.escape(m) for m in markers.values()) + ")"
        parts = re.split(pattern, text)
        marker_to_key = {m: k for k, m in markers.items()}

        content: List[Dict[str, Any]] = []
        for part in parts:
            if part in marker_to_key:
                content.extend(mc_kwargs[marker_to_key[part]].content)
            elif part:
                content.append({"type": "text", "text": part})

        return cls(content=content)

# ============== API Wrappers ==============

ANTHROPIC_FORMAT_PROMPT = """
Please format the final output in the JSON format specified within the <format> tag. Enclose the JSON in <response> tags (i.e., <response>{{JSON-parsable-text}}</response>).
<format>
{format}
</format>
"""


def _split_system_prompt(messages: List[Dict[str, Any]]) -> tuple[List[Dict], str]:
    """Extract system prompt from messages (for Anthropic)."""
    system_prompt = "You are a helpful assistant."
    for message in messages:
        if message["role"] == "system":
            system_prompt = message["content"]
            break

    messages = [message for message in messages if message["role"] != "system"]
    return messages, system_prompt


def get_response_pydantic_openai(
    messages: List[Dict[str, Any]],
    response_format: Any,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
) -> Dict[str, Any]:
    """Call OpenAI API with structured output."""
    if model is None:
        model = DEFAULT_MODEL_OPENAI

    client = _get_openai_client()

    # Build JSON schema from Pydantic model
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
        "max_completion_tokens": get_max_tokens_for_model(model),
        "response_format": response_format_dict,
    }

    # Model-specific parameters
    variable_args = {}
    if "gpt-5" in model:
        variable_args["reasoning_effort"] = REASONING_EFFORT
    else:
        variable_args["temperature"] = temperature if temperature is not None else TEMPERATURE

    try:
        response_raw = client.chat.completions.with_raw_response.create(
            **basic_args,
            **variable_args,
        )
    except BadRequestError as e:
        if "could not parse the json body" in str(e).lower():
            raise LLMParseError(f"OpenAI failed to parse request JSON: {e}") from e
        raise

    completion = response_raw.parse()
    response_text = completion.choices[0].message.content

    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        raise LLMParseError(f"Failed to parse response as JSON: {response_text}")


def get_response_pydantic_anthropic(
    messages: List[Dict[str, Any]],
    response_format: Any,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
) -> Dict[str, Any]:
    """Call Anthropic (via Bedrock) with structured output."""
    if model is None:
        model = DEFAULT_MODEL_ANTHROPIC

    client = _get_anthropic_client()

    # Build JSON schema from Pydantic model
    json_obj = response_format
    if hasattr(response_format, "model_json_schema"):
        json_obj = {
            "schema": response_format.model_json_schema(),
            "name": response_format.__name__,
        }

    response_format_dict = {
        "type": "json_schema",
        "schema": {
            **json_obj,
            "strict": True,
        },
    }

    messages, system_prompt = _split_system_prompt(messages)

    basic_args = {
        "model": model,
        "max_tokens": get_max_tokens_for_model(model),
        "system": system_prompt,
        "messages": messages,
    }

    # Model-specific parameters
    variable_args = {}
    variable_args["output_config"] = {
        "format": response_format_dict,
    }
    if "claude-3" in model:
        variable_args["temperature"] = temperature if temperature is not None else TEMPERATURE
    else:
        variable_args["thinking"] = {
            "type": "adaptive"
        }
        variable_args["output_config"] = {
            **variable_args["output_config"],
            "effort": REASONING_EFFORT
        }

    completion = client.messages.create(
        **basic_args,
        **variable_args,
    )

    response_text = completion.content[0].text

    try:
        return json.loads(response_text)
    except (json.JSONDecodeError, IndexError):
        raise LLMParseError(f"Failed to extract JSON from response: {response_text}")


def get_response_pydantic_gemini(
    messages: List[Dict[str, Any]],
    response_format: Any,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
) -> Dict[str, Any]:
    """Call Gemini API with structured output."""
    if model is None:
        model = DEFAULT_MODEL_GEMINI

    client = _get_gemini_client()

    # Build JSON schema (accept both pydantic class and pre-serialized dict)
    if hasattr(response_format, "model_json_schema"):
        schema = response_format.model_json_schema()
    elif isinstance(response_format, dict) and "schema" in response_format:
        schema = response_format["schema"]
    else:
        schema = response_format

    # Split system prompt from messages
    messages, system_prompt = _split_system_prompt(list(messages))

    # Convert messages to Gemini Content format
    role_map = {"user": "user", "assistant": "model"}
    contents: List[genai_types.Content] = []
    for msg in messages:
        role = role_map.get(msg["role"], "user")
        content = msg["content"]
        if isinstance(content, str):
            parts = [genai_types.Part.from_text(text=content)]
        else:
            parts = list(content)
        contents.append(genai_types.Content(role=role, parts=parts))

    basic_args = {
        "system_instruction": system_prompt,
        "seed": SEED,
        "max_output_tokens": get_max_tokens_for_model(model),
        "response_mime_type": "application/json",
        "response_json_schema": schema,
        "automatic_function_calling": genai_types.AutomaticFunctionCallingConfig(disable=True),
    }

    ## Model-specific parameters
    variable_args = {}
    if "gemini-3" in model:
        variable_args["thinking_config"] = genai_types.ThinkingConfig(thinking_level=REASONING_EFFORT)
    if temperature is not None:
        variable_args["temperature"] = temperature

    config = genai_types.GenerateContentConfig(
        **basic_args,
        **variable_args
    )

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )

    response_text = response.text

    try:
        return json.loads(response_text)
    except (json.JSONDecodeError, TypeError):
        raise LLMParseError(f"Failed to parse response as JSON: {response_text}")

# ============== Batch Processing ==============


def random_uid() -> str:
    """Generate a random request ID."""
    return str(uuid4())


def _ensure_parent_dir(path: str | Path) -> None:
    """Ensure the parent directory exists."""
    parent = Path(path).parent
    if parent and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)


def llm_get_requests_path(batch_id: str) -> str:
    """Get path to requests file for a batch."""
    return os.path.join(BATCH_PATH, f"{batch_id}_llm_requests.jsonl")


def llm_get_responses_path(batch_id: str) -> str:
    """Get path to responses file for a batch."""
    return os.path.join(BATCH_PATH, f"{batch_id}_llm_responses.jsonl")


_append_lock = threading.Lock()


def _append_jsonl(path: str, obj: Any) -> None:
    """Thread-safe append to JSONL file."""
    line = json.dumps(obj, ensure_ascii=False)
    with _append_lock:
        _ensure_parent_dir(path)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def llm_enqueue_request(
    batch_id: str,
    messages: List[Dict[str, Any]],
    response_format: Any,
    options: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None,
) -> str:
    """
    Enqueue a request for batch processing.

    Returns:
        request_id that can be used to retrieve the response later
    """
    req_id = request_id or random_uid()
    response_format_dict = {
        "schema": response_format.model_json_schema(),
        "name": response_format.__name__,
    }

    # Convert MessageContent objects to dicts for JSON serialization
    serializable_messages = []
    for msg in messages:
        content = msg["content"]
        if isinstance(content, MessageContent):
            content = content.to_dict()
        serializable_messages.append({"role": msg["role"], "content": content})

    record = {
        "request_id": req_id,
        "messages": serializable_messages,
        "response_format": response_format_dict,
        "options": options or {},
    }
    path = llm_get_requests_path(batch_id)
    _append_jsonl(path, record)
    return req_id


def llm_get_response(batch_id: str, request_id: str) -> Dict[str, Any]:
    """Get a specific response by request ID."""
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
                    raise ValueError(f"Request failed with errors: {result['errors']}")
                return result
    raise ValueError(f"Request ID not found: {request_id}")


def llm_get_responses(batch_id: str) -> Dict[str, Dict[str, Any]]:
    """Get all responses for a batch as a dict mapping request_id -> response data."""
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
                raise ValueError(f"Request failed with errors: {result['errors']}")
            # Return both options (from request) and response (from result)
            responses[request_json["request_id"]] = {
                "options": request_json.get("options", {}),
                "response": result
            }
    return responses


# ============== Rate Limiting ==============


class TokenBucket:
    """Token bucket for rate limiting."""

    def __init__(self, max_per_minute: float):
        self.capacity = float(max_per_minute)
        self.tokens = float(max_per_minute)
        self.rate_per_sec = float(max_per_minute) / 60.0
        self.last = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self, amount: float = 1.0):
        """Acquire tokens, waiting if necessary."""
        while True:
            async with self._lock:
                now = time.time()
                elapsed = now - self.last
                if elapsed > 0:
                    self.tokens = min(
                        self.capacity, self.tokens + elapsed * self.rate_per_sec
                    )
                    self.last = now
                if self.tokens >= amount:
                    self.tokens -= amount
                    return
                needed = amount - self.tokens
                wait_secs = needed / self.rate_per_sec if self.rate_per_sec > 0 else 0.05
            await asyncio.sleep(wait_secs)


# ============== Error Handling ==============


def _is_rate_limit_error(err: Exception) -> bool:
    """Check if error is a rate limit error."""
    msg = str(err).lower()
    return (
        "rate limit" in msg
        or "429" in msg
        or "too many requests" in msg
        or "throttl" in msg
    )


def _is_retryable_http_error(err: Exception) -> bool:
    """Check if error is retryable (network/server issues)."""
    msg = str(err).lower()
    return any(
        k in msg
        for k in [
            "timeout",
            "temporar",
            "reset",
            "connection",
            "server error",
            "503",
            "502",
            "504",
            "refusal",
            # Gemini occasionally returns transient 403 SERVICE_DISABLED even
            # on a properly enabled project; a retry with backoff resolves it.
            "service_disabled",
        ]
    )


def _classify_error(err: Exception) -> Literal["rate", "retryable", "fatal"]:
    """Classify error for retry logic."""
    if _is_rate_limit_error(err):
        return "rate"
    if isinstance(err, LLMParseError):
        return "retryable"
    if _is_retryable_http_error(err):
        return "retryable"
    return "fatal"


def _backoff_seconds(attempt_idx: int, base: float = 1.0, cap: float = 60.0) -> float:
    """Calculate exponential backoff with jitter."""
    expo = base * (2 ** (attempt_idx - 1))
    with_jitter = expo * (0.5 + random.random())  # 0.5x–1.5x jitter
    return min(with_jitter, cap)


# ============== Status Tracking ==============


@dataclass
class StatusTracker:
    """Track batch processing status."""

    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0
    num_other_errors: int = 0
    time_of_last_rate_limit_error: float = 0.0


@dataclass
class APIRequest:
    """Request envelope for batch processing."""

    task_id: int
    request_json: dict
    attempts_left: int
    token_consumption: int


# ============== Token Estimation ==============


def _encoding_for(model: str | None):
    """Get tiktoken encoding for a model."""
    try:
        if model:
            return tiktoken.encoding_for_model(model)
    except Exception:
        pass
    return tiktoken.get_encoding("cl100k_base")


def _count_text_tokens(text: str, enc) -> int:
    """Count tokens in text."""
    if not text:
        return 0
    return len(enc.encode(text))


def _extract_text_from_content(content: Any) -> List[str]:
    """Extract text portions from content (ignoring images)."""
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


def estimate_prompt_tokens_from_messages(
    messages: List[Dict[str, Any]], model: str | None
) -> int:
    """Estimate token count for a list of messages."""
    enc = _encoding_for(model)
    total = 0
    for msg in messages or []:
        total += MESSAGE_OVERHEAD_TOKENS
        name = msg.get("name")
        if name:
            total -= 1  # role omission

        # Extract and count text
        contents = _extract_text_from_content(msg.get("content"))
        for t in contents:
            total += _count_text_tokens(t, enc)

    total += REPLY_PRIMING_TOKENS
    return total


def estimate_token_consumption(request_json: dict, model: str | None) -> int:
    """Estimate total token consumption for a request."""
    messages = request_json.get("messages", [])
    prompt_tokens = estimate_prompt_tokens_from_messages(messages, model)

    # Add Anthropic formatting overhead if applicable
    if model and model.startswith("us.anthropic."):
        prompt_tokens += estimate_prompt_tokens_from_messages(
            [{"role": "user", "content": ANTHROPIC_FORMAT_PROMPT}], model
        )

    return prompt_tokens


# ============== Parallel Processing ==============


async def process_api_requests(
    batch_id: str,
    model_config: ModelConfig,
    max_concurrency: int = 64,
    seconds_to_pause_after_limit_error: float = 60.0,
    temperature: Optional[float] = None,
):
    """
    Process all enqueued requests in parallel with rate limiting and retries.

    Args:
        batch_id: Batch identifier
        max_concurrency: Maximum concurrent requests
        seconds_to_pause_after_limit_error: Pause duration after rate limit errors
    """
    requests_path = llm_get_requests_path(batch_id)
    responses_path = llm_get_responses_path(batch_id)

    # Select the appropriate call function
    if model_config.provider == "openai":
        call_function = get_response_pydantic_openai
    elif model_config.provider == "anthropic":
        call_function = get_response_pydantic_anthropic
    elif model_config.provider == "gemini":
        call_function = get_response_pydantic_gemini
    else:
        raise ValueError(f"Unknown provider: {model_config.provider}")

    # Rate limiting
    req_bucket = TokenBucket(MAX_REQUESTS_PER_MINUTE)
    tok_bucket = TokenBucket(MAX_TOKENS_PER_MINUTE)

    status = StatusTracker()
    queue = asyncio.Queue(maxsize=max_concurrency * 4)

    # Producer: read requests from JSONL and enqueue
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
                    token_consumption=estimate_token_consumption(record, model_config.model),
                )
                task_id += 1
                status.num_tasks_started += 1
                status.num_tasks_in_progress += 1
                await queue.put(req)
        # Signal workers to stop
        for _ in range(max_concurrency):
            await queue.put(None)

    # Worker: process requests with retries
    async def worker(worker_id: int):
        nonlocal status
        while True:
            req = await queue.get()
            if req is None:
                return

            # Throttle based on rate limits
            await req_bucket.acquire(1.0)
            await tok_bucket.acquire(float(req.token_consumption))

            attempt = 0
            errors = []
            while True:
                try:

                    def _call():
                        r = req.request_json
                        messages = r.get("messages")
                        response_format = r.get("response_format")

                        # Reconstruct MessageContent from serialized dicts
                        processed_messages = []
                        for msg in messages:
                            content = msg["content"]
                            # Check if content is a serialized MessageContent
                            if isinstance(content, dict) and "content" in content and isinstance(content["content"], list):
                                # Reconstruct MessageContent and convert to API format
                                msg_content = MessageContent.from_dict(content)
                                if model_config.provider == "openai":
                                    content = msg_content.to_openai_content()
                                elif model_config.provider == "gemini":
                                    content = msg_content.to_gemini_content()
                                else:
                                    content = msg_content.to_anthropic_content()
                            processed_messages.append({"role": msg["role"], "content": content})

                        return call_function(processed_messages, response_format, model_config.model, temperature=temperature)

                    result = await asyncio.to_thread(_call)
                    _append_jsonl(responses_path, [req.request_json, result])
                    status.num_tasks_succeeded += 1
                    status.num_tasks_in_progress -= 1
                    break
                except Exception as e:
                    errors.append(str(e))
                    kind = _classify_error(e)

                    if kind == "rate":
                        logger.warning(
                            "Rate limit error (attempt %d, request %s): %s — pausing %.0fs",
                            attempt + 1, req.request_json.get("request_id", "?"), e, seconds_to_pause_after_limit_error,
                        )
                        status.num_rate_limit_errors += 1
                        status.time_of_last_rate_limit_error = time.time()
                        await asyncio.sleep(seconds_to_pause_after_limit_error)
                    elif kind == "retryable":
                        logger.warning(
                            "Retryable error (attempt %d, request %s): %s",
                            attempt + 1, req.request_json.get("request_id", "?"), e,
                        )
                        status.num_api_errors += 1
                    else:
                        logger.warning(
                            "Fatal error (attempt %d, request %s): %s — not retrying",
                            attempt + 1, req.request_json.get("request_id", "?"), e,
                        )
                        status.num_other_errors += 1
                        _append_jsonl(
                            responses_path, [req.request_json, {"errors": errors}]
                        )
                        status.num_tasks_failed += 1
                        status.num_tasks_in_progress -= 1
                        break

                    # Check if we should retry
                    req.attempts_left -= 1
                    attempt += 1
                    if req.attempts_left <= 0:
                        logger.warning(
                            "All attempts exhausted (request %s), errors: %s",
                            req.request_json.get("request_id", "?"), errors,
                        )
                        _append_jsonl(
                            responses_path, [req.request_json, {"errors": errors}]
                        )
                        status.num_tasks_failed += 1
                        status.num_tasks_in_progress -= 1
                        break

                    # Backoff before retry
                    await asyncio.sleep(_backoff_seconds(attempt))

    # Run producer and workers
    await asyncio.gather(producer(), *[worker(i) for i in range(max_concurrency)])

    logger.info(
        f"Batch {batch_id} complete: "
        f"Success={status.num_tasks_succeeded} "
        f"Failed={status.num_tasks_failed} "
        f"RateLimitErrors={status.num_rate_limit_errors}"
    )

def batch_run_llm_calls(
    batch_id: str,
    request_args,
    func_request,
    func_response,
    max_concurrency: int,
    model_config: ModelConfig,
    temperature: Optional[float] = None,
):
    logger.info(f"Batch ID for {func_request.__name__}: {batch_id}")
    if len(request_args) == 0:
        return []
    req_ids = []
    for request_arg in request_args:
        api_args = func_request(**request_arg)
        req_id = llm_enqueue_request(batch_id, **api_args)
        req_ids.append(req_id)

    asyncio.run(process_api_requests(
        batch_id,
        max_concurrency=max_concurrency,
        model_config=model_config,
        temperature=temperature,
    ))
    batch_responses = llm_get_responses(batch_id)

    batch_results = []
    for request_arg, req_id in zip(request_args, req_ids):
        if req_id not in batch_responses:
            raise ValueError(f"Request ID not found: {req_id}")
        response = batch_responses[req_id]
        result = func_response(response=response, **request_arg)
        batch_results.append(result)
    return batch_results