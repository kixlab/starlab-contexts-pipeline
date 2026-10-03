"""Multi-turn tool calling: provider adapters round-trip transcripts (no network)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from anthropic.types import Message
from google.genai import types as genai_types
from openai.types.chat import ChatCompletion

from utils import ModelConfig
from utils import llm as llm_mod
from utils import tool_loop
from utils.tool_loop import ToolLoopClient, ToolSpec

TOOLS = [ToolSpec(name="read_video", description="Read one tutorial.",
                  parameters={"type": "object",
                              "properties": {"video_id": {"type": "string"}},
                              "required": ["video_id"]})]


def _transcript_after(turn):
    """User prompt, the assistant's tool call, and the tool's result."""
    call = turn.tool_calls[0]
    return [
        {"role": "user", "text": "Propose a factor."},
        turn.message,
        {"role": "tool", "tool_call_id": call.id, "name": call.name, "content": "segments..."},
    ]


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------

class FakeGemini:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.models = self

    def generate_content(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        return self.responses.pop(0)


def _gemini_response(parts, usage=(100, 10, 5)):
    return genai_types.GenerateContentResponse(
        candidates=[genai_types.Candidate(content=genai_types.Content(role="model", parts=parts))],
        usage_metadata=genai_types.GenerateContentResponseUsageMetadata(
            prompt_token_count=usage[0], candidates_token_count=usage[1],
            thoughts_token_count=usage[2]),
    )


def test_gemini_round_trip_keeps_thought_signature(monkeypatch):
    fake = FakeGemini([
        _gemini_response([genai_types.Part(
            function_call=genai_types.FunctionCall(id="c1", name="read_video",
                                                   args={"video_id": "v1"}),
            thought_signature=b"sig-1")]),
        _gemini_response([genai_types.Part(text="Done.")]),
    ])
    monkeypatch.setattr(llm_mod, "_get_gemini_client", lambda: fake)
    client = ToolLoopClient(ModelConfig("vlm", "gemini", "gemini-3-flash-preview",
                                        reasoning_effort="low"))

    first = client.step("SYSTEM", [{"role": "user", "text": "Propose a factor."}], TOOLS, seed=7)
    assert [(c.id, c.name, c.arguments) for c in first.tool_calls] == [
        ("c1", "read_video", {"video_id": "v1"})]
    assert first.usage == {"input_tokens": 100, "output_tokens": 10, "thinking_tokens": 5}

    second = client.step("SYSTEM", _transcript_after(first), TOOLS, seed=8)
    assert second.text == "Done." and second.tool_calls == []

    contents = fake.calls[1]["contents"]
    assert [c.role for c in contents] == ["user", "model", "user"]
    assert contents[1].parts[0].thought_signature == b"sig-1"
    response = contents[2].parts[0].function_response
    assert (response.id, response.name, response.response) == (
        "c1", "read_video", {"result": "segments..."})

    config = fake.calls[0]["config"]
    assert config.seed == 7 and config.system_instruction == "SYSTEM"
    assert config.thinking_config.thinking_level.value.lower() == "low"
    decl = config.tools[0].function_declarations[0]
    assert decl.name == "read_video" and decl.parameters_json_schema == TOOLS[0].parameters


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

def _openai_completion(message, usage=(200, 30, 12)):
    return ChatCompletion.model_validate({
        "id": "x", "object": "chat.completion", "created": 0, "model": "gpt-5",
        "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
        "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1],
                  "total_tokens": usage[0] + usage[1],
                  "completion_tokens_details": {"reasoning_tokens": usage[2]}},
    })


def test_openai_round_trip(monkeypatch):
    calls = []
    responses = [
        _openai_completion({"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "read_video", "arguments": '{"video_id": "v2"}'}}]}),
        _openai_completion({"role": "assistant", "content": "Done."}),
    ]
    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kw: calls.append(kw) or responses.pop(0))))
    monkeypatch.setattr(llm_mod, "_get_openai_client", lambda: fake)
    client = ToolLoopClient(ModelConfig("vlm", "openai", "gpt-5"))

    first = client.step("SYSTEM", [{"role": "user", "text": "Propose a factor."}], TOOLS, seed=3)
    assert [(c.id, c.arguments) for c in first.tool_calls] == [("call_1", {"video_id": "v2"})]
    assert first.usage == {"input_tokens": 200, "output_tokens": 18, "thinking_tokens": 12}

    client.step("SYSTEM", _transcript_after(first), TOOLS)
    msgs = calls[1]["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "tool"]
    assert msgs[2]["tool_calls"][0]["id"] == "call_1"
    assert msgs[3] == {"role": "tool", "tool_call_id": "call_1", "content": "segments..."}
    assert calls[0]["seed"] == 3 and calls[0]["tools"][0]["function"]["name"] == "read_video"
    assert calls[0]["reasoning_effort"] == llm_mod.REASONING_EFFORT


def test_openai_invalid_json_arguments_are_surfaced(monkeypatch):
    bad = _openai_completion({"role": "assistant", "content": None, "tool_calls": [
        {"id": "c", "type": "function", "function": {"name": "read_video", "arguments": "{oops"}}]})
    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: bad)))
    monkeypatch.setattr(llm_mod, "_get_openai_client", lambda: fake)
    turn = ToolLoopClient(ModelConfig("vlm", "openai", "gpt-5")).step("S", [], TOOLS)
    assert turn.tool_calls[0].arguments == {"__invalid_json__": "{oops"}


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

def _anthropic_message(content, usage=(300, 40)):
    return Message.model_validate({
        "id": "m", "type": "message", "role": "assistant", "model": "claude",
        "content": content, "stop_reason": "tool_use", "stop_sequence": None,
        "usage": {"input_tokens": usage[0], "output_tokens": usage[1]},
    })


def test_anthropic_round_trip_keeps_thinking_blocks(monkeypatch):
    calls = []
    responses = [
        _anthropic_message([
            {"type": "thinking", "thinking": "Let me look.", "signature": "sig-a"},
            {"type": "tool_use", "id": "tu_1", "name": "read_video", "input": {"video_id": "v1"}},
        ]),
        _anthropic_message([{"type": "text", "text": "Done."}]),
    ]
    fake = SimpleNamespace(messages=SimpleNamespace(
        create=lambda **kw: calls.append(kw) or responses.pop(0)))
    monkeypatch.setattr(llm_mod, "_get_anthropic_client", lambda: fake)
    client = ToolLoopClient(ModelConfig("vlm", "anthropic", "claude-opus-4"))

    first = client.step("SYSTEM", [{"role": "user", "text": "Propose a factor."}], TOOLS)
    assert [(c.id, c.arguments) for c in first.tool_calls] == [("tu_1", {"video_id": "v1"})]
    assert first.usage == {"input_tokens": 300, "output_tokens": 40, "thinking_tokens": 0}

    client.step("SYSTEM", _transcript_after(first), TOOLS)
    msgs = calls[1]["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert msgs[1]["content"][0] == {"type": "thinking", "thinking": "Let me look.",
                                     "signature": "sig-a"}
    assert msgs[2]["content"] == [{"type": "tool_result", "tool_use_id": "tu_1",
                                   "content": "segments..."}]
    assert calls[0]["tools"][0]["input_schema"] == TOOLS[0].parameters
    assert calls[0]["system"] == "SYSTEM"


# ---------------------------------------------------------------------------
# Retries and usage
# ---------------------------------------------------------------------------

def test_transient_errors_are_retried_and_usage_recorded(monkeypatch):
    attempts = []

    def flaky(model, contents, config):
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("503 server error")
        return _gemini_response([genai_types.Part(text="ok")], usage=(50, 5, 1))

    monkeypatch.setattr(llm_mod, "_get_gemini_client",
                        lambda: SimpleNamespace(models=SimpleNamespace(generate_content=flaky)))
    monkeypatch.setattr(tool_loop.time, "sleep", lambda s: None)
    before = llm_mod.USAGE.snapshot()
    with llm_mod.usage_stage("agent"):
        turn = ToolLoopClient(ModelConfig("vlm", "gemini", "gemini-3-flash-preview")).step(
            "S", [{"role": "user", "text": "hi"}], TOOLS)
    assert turn.text == "ok" and len(attempts) == 2
    assert llm_mod.usage_delta(llm_mod.USAGE.snapshot(), before) == {
        "agent": {"input_tokens": 50, "output_tokens": 5, "thinking_tokens": 1, "requests": 1}}


def test_fatal_errors_are_not_retried(monkeypatch):
    attempts = []

    def broken(model, contents, config):
        attempts.append(1)
        raise ValueError("400 invalid argument")

    monkeypatch.setattr(llm_mod, "_get_gemini_client",
                        lambda: SimpleNamespace(models=SimpleNamespace(generate_content=broken)))
    with pytest.raises(ValueError):
        ToolLoopClient(ModelConfig("vlm", "gemini", "gemini-3-flash-preview")).step("S", [], TOOLS)
    assert len(attempts) == 1
