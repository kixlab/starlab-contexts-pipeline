"""
Multi-turn tool calling for Gemini, OpenAI, and Anthropic (Bedrock).

Conversations are kept as a provider-neutral list of message dicts:

    {"role": "user", "text": str}
    {"role": "assistant", "text": str, "tool_calls": [{"id", "name", "arguments"}],
     "raw": <provider-native assistant content>}
    {"role": "tool", "tool_call_id": str, "name": str, "content": str}

``raw`` is replayed verbatim on the next turn, so provider requirements such as
Gemini thought signatures and Anthropic thinking blocks are preserved; a
transcript is therefore tied to the provider that produced it. Tool messages
hold plain text, so callers can trim old tool outputs without knowing the
provider.

``ToolLoopClient.step`` sends one turn and returns the assistant's reply. It
retries like batch requests (rate limits, transient errors) and raises once
attempts are exhausted. Token usage is added to ``utils.llm.USAGE`` under the
current ``usage_stage``.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from google.genai import types as genai_types

from . import llm
from .config import ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class ToolSpec:
    """A callable tool: name, description, and JSON-schema parameters."""

    name: str
    description: str
    parameters: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "description": self.description,
                "parameters": self.parameters}


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class AgentTurn:
    """One assistant reply: the transcript message plus token usage."""

    message: Dict[str, Any]
    usage: Dict[str, int] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return self.message.get("text", "")

    @property
    def tool_calls(self) -> List[ToolCall]:
        return [ToolCall(**c) for c in self.message.get("tool_calls", [])]


class ToolLoopClient:
    """Sends a transcript plus tool specs to the configured model."""

    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        steps = {"gemini": _gemini_step, "openai": _openai_step, "anthropic": _anthropic_step}
        if model_config.provider not in steps:
            raise ValueError(f"Unsupported provider: {model_config.provider}")
        self._step = steps[model_config.provider]

    def step(
        self,
        system: str,
        messages: List[Dict[str, Any]],
        tools: List[ToolSpec],
        seed: Optional[int] = None,
    ) -> AgentTurn:
        turn = _with_retries(
            lambda: self._step(self.model_config, system, messages, tools, seed)
        )
        if turn.usage:
            llm.USAGE.add(llm._usage_stage.get(), turn.usage)
        return turn


def _with_retries(call: Callable[[], AgentTurn]) -> AgentTurn:
    for attempt in range(1, llm.MAX_ATTEMPTS + 1):
        try:
            return call()
        except Exception as e:
            kind = llm._classify_error(e)
            if kind == "fatal" or attempt == llm.MAX_ATTEMPTS:
                raise
            pause = 60.0 if kind == "rate" else llm._backoff_seconds(attempt)
            logger.warning("Tool-loop %s error (attempt %d): %s; retrying in %.0fs",
                           kind, attempt, e, pause)
            time.sleep(pause)
    raise AssertionError("unreachable")


def _effort(model_config: ModelConfig) -> str:
    return model_config.reasoning_effort or llm.REASONING_EFFORT


def _group_tool_messages(messages: List[Dict[str, Any]]) -> List[Any]:
    """Merge consecutive user/tool messages into one user turn (list of messages)."""
    grouped: List[Any] = []
    for m in messages:
        if m["role"] == "assistant":
            grouped.append(m)
        elif grouped and isinstance(grouped[-1], list):
            grouped[-1].append(m)
        else:
            grouped.append([m])
    return grouped


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------

def _gemini_step(model_config, system, messages, tools, seed) -> AgentTurn:
    contents: List[genai_types.Content] = []
    for entry in _group_tool_messages(messages):
        if isinstance(entry, dict):  # assistant
            contents.append(genai_types.Content.model_validate(entry["raw"]))
            continue
        parts = []
        for m in entry:
            if m["role"] == "tool":
                parts.append(genai_types.Part(function_response=genai_types.FunctionResponse(
                    id=m["tool_call_id"], name=m["name"], response={"result": m["content"]},
                )))
            else:
                parts.append(genai_types.Part.from_text(text=m["text"]))
        contents.append(genai_types.Content(role="user", parts=parts))

    config_args = dict(
        system_instruction=system,
        max_output_tokens=llm.get_max_tokens_for_model(model_config.model),
        tools=[genai_types.Tool(function_declarations=[
            genai_types.FunctionDeclaration(
                name=t.name, description=t.description, parameters_json_schema=t.parameters,
            ) for t in tools
        ])],
        automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(disable=True),
    )
    if seed is not None:
        config_args["seed"] = seed
    if "gemini-3" in model_config.model:
        config_args["thinking_config"] = genai_types.ThinkingConfig(
            thinking_level=_effort(model_config),
        )
    response = llm._get_gemini_client().models.generate_content(
        model=model_config.model, contents=contents,
        config=genai_types.GenerateContentConfig(**config_args),
    )

    content = response.candidates[0].content if response.candidates else None
    parts = (content.parts or []) if content is not None else []
    text = "".join(p.text for p in parts if p.text and not p.thought)
    calls = [
        {"id": p.function_call.id or f"call_{i}", "name": p.function_call.name,
         "arguments": dict(p.function_call.args or {})}
        for i, p in enumerate(parts) if p.function_call is not None
    ]
    raw = (content.model_dump(mode="json", exclude_none=True) if content is not None
           else {"role": "model", "parts": [{"text": ""}]})
    usage = response.usage_metadata
    return AgentTurn(
        message={"role": "assistant", "text": text, "tool_calls": calls, "raw": raw},
        usage={
            "input_tokens": getattr(usage, "prompt_token_count", 0) or 0,
            "output_tokens": getattr(usage, "candidates_token_count", 0) or 0,
            "thinking_tokens": getattr(usage, "thoughts_token_count", 0) or 0,
        } if usage is not None else {},
    )


# ---------------------------------------------------------------------------
# OpenAI (chat completions)
# ---------------------------------------------------------------------------

def _openai_step(model_config, system, messages, tools, seed) -> AgentTurn:
    msgs: List[Dict[str, Any]] = [{"role": "system", "content": system}]
    for m in messages:
        if m["role"] == "user":
            msgs.append({"role": "user", "content": m["text"]})
        elif m["role"] == "assistant":
            msgs.append(m["raw"])
        else:
            msgs.append({"role": "tool", "tool_call_id": m["tool_call_id"],
                         "content": m["content"]})

    args: Dict[str, Any] = dict(
        model=model_config.model,
        messages=msgs,
        tools=[{"type": "function", "function": t.to_dict()} for t in tools],
        max_completion_tokens=llm.get_max_tokens_for_model(model_config.model),
        seed=llm.SEED if seed is None else seed,
    )
    if "gpt-5" in model_config.model:
        args["reasoning_effort"] = _effort(model_config)
    else:
        args["temperature"] = llm.TEMPERATURE
    completion = llm._get_openai_client().chat.completions.create(**args)

    message = completion.choices[0].message
    calls = []
    for c in message.tool_calls or []:
        try:
            arguments = json.loads(c.function.arguments or "{}")
        except json.JSONDecodeError:
            arguments = {"__invalid_json__": c.function.arguments}
        calls.append({"id": c.id, "name": c.function.name, "arguments": arguments})
    raw: Dict[str, Any] = {"role": "assistant", "content": message.content or ""}
    if message.tool_calls:
        raw["tool_calls"] = [c.model_dump(exclude_none=True) for c in message.tool_calls]
    usage = completion.usage
    usage_dict: Dict[str, int] = {}
    if usage is not None:
        details = getattr(usage, "completion_tokens_details", None)
        reasoning = getattr(details, "reasoning_tokens", 0) or 0
        usage_dict = {"input_tokens": usage.prompt_tokens,
                      "output_tokens": usage.completion_tokens - reasoning,
                      "thinking_tokens": reasoning}
    return AgentTurn(
        message={"role": "assistant", "text": message.content or "", "tool_calls": calls,
                 "raw": raw},
        usage=usage_dict,
    )


# ---------------------------------------------------------------------------
# Anthropic (Bedrock)
# ---------------------------------------------------------------------------

def _anthropic_step(model_config, system, messages, tools, seed) -> AgentTurn:
    msgs: List[Dict[str, Any]] = []
    for entry in _group_tool_messages(messages):
        if isinstance(entry, dict):  # assistant
            msgs.append({"role": "assistant", "content": entry["raw"]})
            continue
        blocks = []
        for m in entry:
            if m["role"] == "tool":
                blocks.append({"type": "tool_result", "tool_use_id": m["tool_call_id"],
                               "content": m["content"]})
            else:
                blocks.append({"type": "text", "text": m["text"]})
        msgs.append({"role": "user", "content": blocks})

    args: Dict[str, Any] = dict(
        model=model_config.model,
        max_tokens=llm.get_max_tokens_for_model(model_config.model),
        system=system,
        messages=msgs,
        tools=[{"name": t.name, "description": t.description, "input_schema": t.parameters}
               for t in tools],
    )
    if "claude-3" in model_config.model:
        args["temperature"] = llm.TEMPERATURE
    else:
        args["thinking"] = {"type": "adaptive"}
        args["output_config"] = {"effort": _effort(model_config)}
    completion = llm._get_anthropic_client().messages.create(**args)

    text = "".join(b.text for b in completion.content if b.type == "text")
    calls = [{"id": b.id, "name": b.name, "arguments": dict(b.input or {})}
             for b in completion.content if b.type == "tool_use"]
    raw = [b.model_dump(exclude_none=True) for b in completion.content]
    usage = getattr(completion, "usage", None)
    return AgentTurn(
        message={"role": "assistant", "text": text, "tool_calls": calls, "raw": raw},
        # Thinking tokens are included in output_tokens and not reported apart.
        usage={"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
               "thinking_tokens": 0} if usage is not None else {},
    )
