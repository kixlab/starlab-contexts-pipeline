"""batch_run_llm_calls: per-request seeds reach the provider call (no network)."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from utils import ModelConfig
from utils import llm as llm_mod


class Echo(BaseModel):
    value: int


def _form(value: int, **kwargs):
    return {
        "messages": [{"role": "user", "content": f"value {value}"}],
        "response_format": Echo,
    }


def _process(response, **kwargs):
    return response["response"]


@pytest.fixture
def recorded(tmp_path, monkeypatch):
    calls = []

    def fake_gemini(messages, response_format, model=None, temperature=None, seed=None,
                    reasoning_effort=None):
        calls.append({"text": messages[0]["content"], "seed": seed,
                      "temperature": temperature})
        return {"value": int(messages[0]["content"].split()[-1])}

    monkeypatch.setattr(llm_mod, "BATCH_PATH", str(tmp_path) + "/")
    monkeypatch.setattr(llm_mod, "get_response_pydantic_gemini", fake_gemini)
    return calls


def _run(seeds=None, temperature=None):
    return llm_mod.batch_run_llm_calls(
        f"test_{llm_mod.random_uid()}",
        [{"value": v} for v in (1, 2, 3)],
        _form, _process,
        max_concurrency=4,
        model_config=ModelConfig(category="vlm", provider="gemini", model="fake"),
        temperature=temperature,
        seeds=seeds,
    )


def test_per_request_seeds_reach_provider(recorded):
    results = _run(seeds=[11, None, 13], temperature=0.5)
    assert [r["value"] for r in results] == [1, 2, 3]
    by_text = {c["text"]: c for c in recorded}
    assert by_text["value 1"]["seed"] == 11
    assert by_text["value 2"]["seed"] is None  # provider default (SEED)
    assert by_text["value 3"]["seed"] == 13
    assert {c["temperature"] for c in recorded} == {0.5}


def test_no_seeds_means_provider_default(recorded):
    _run()
    assert {c["seed"] for c in recorded} == {None}


def test_misaligned_seeds_rejected(recorded):
    with pytest.raises(ValueError):
        _run(seeds=[1, 2])


# ---- permanent failures and outages ----

@pytest.fixture
def flaky(tmp_path, monkeypatch):
    """Provider whose behavior per request value is set by the test."""
    behavior = {}

    def fake_gemini(messages, response_format, model=None, temperature=None, seed=None,
                    reasoning_effort=None):
        value = int(messages[0]["content"].split()[-1])
        kind = behavior.get(value, "ok")
        if kind == "parse":
            raise llm_mod.LLMParseError("not json")
        if kind == "service":
            raise RuntimeError("503 server error")
        return {"value": value, llm_mod.USAGE_KEY: {
            "input_tokens": 100, "output_tokens": 10, "thinking_tokens": 5}}

    monkeypatch.setattr(llm_mod, "BATCH_PATH", str(tmp_path) + "/")
    monkeypatch.setattr(llm_mod, "get_response_pydantic_gemini", fake_gemini)
    monkeypatch.setattr(llm_mod, "MAX_ATTEMPTS", 2)
    monkeypatch.setattr(llm_mod, "_backoff_seconds", lambda attempt: 0)
    return behavior


def _run_flaky(allow_errors):
    return llm_mod.batch_run_llm_calls(
        f"test_{llm_mod.random_uid()}",
        [{"value": v} for v in (1, 2, 3)],
        _form, _process,
        max_concurrency=4,
        model_config=ModelConfig(category="vlm", provider="gemini", model="fake"),
        allow_errors=allow_errors,
    )


def test_failed_request_raises_by_default(flaky):
    flaky[2] = "parse"
    with pytest.raises(ValueError):
        _run_flaky(allow_errors=False)


def test_allow_errors_returns_failure_markers(flaky):
    flaky[2] = "parse"
    flaky[3] = "service"
    results = _run_flaky(allow_errors=True)
    assert results[0] == {"value": 1}  # usage stripped before processing
    assert isinstance(results[1], llm_mod.LLMRequestFailed) and results[1].kind == "request"
    assert isinstance(results[2], llm_mod.LLMRequestFailed) and results[2].kind == "service"


def test_all_service_failures_are_an_outage(flaky):
    flaky.update({1: "service", 2: "service", 3: "parse"})
    with pytest.raises(llm_mod.LLMBatchOutage):
        _run_flaky(allow_errors=True)


def test_all_request_failures_are_not_an_outage(flaky):
    flaky.update({1: "parse", 2: "parse", 3: "parse"})
    results = _run_flaky(allow_errors=True)
    assert all(isinstance(r, llm_mod.LLMRequestFailed) for r in results)


def test_usage_is_attributed_to_the_current_stage(flaky):
    flaky[2] = "parse"
    before = llm_mod.USAGE.snapshot()
    with llm_mod.usage_stage("labeling"):
        _run_flaky(allow_errors=True)
    delta = llm_mod.usage_delta(llm_mod.USAGE.snapshot(), before)
    assert delta == {"labeling": {"input_tokens": 200, "output_tokens": 20,
                                  "thinking_tokens": 10, "requests": 2}}


# ---- per-model reasoning effort ----

def test_model_reasoning_effort_reaches_provider(tmp_path, monkeypatch):
    seen = []

    def fake_gemini(messages, response_format, model=None, temperature=None, seed=None,
                    reasoning_effort=None):
        seen.append(reasoning_effort)
        return {"value": 1}

    monkeypatch.setattr(llm_mod, "BATCH_PATH", str(tmp_path) + "/")
    monkeypatch.setattr(llm_mod, "get_response_pydantic_gemini", fake_gemini)
    llm_mod.batch_run_llm_calls(
        "test_effort", [{"value": 1}], _form, _process, max_concurrency=1,
        model_config=ModelConfig(category="vlm", provider="gemini", model="fake",
                                 reasoning_effort="low"),
    )
    assert seen == ["low"]


def test_unset_reasoning_effort_keeps_model_identity():
    plain = ModelConfig(category="vlm", provider="gemini", model="m")
    assert plain.to_dict() == {"category": "vlm", "provider": "gemini", "model": "m"}
    low = ModelConfig.from_dict({**plain.to_dict(), "reasoning_effort": "low"})
    assert low.to_dict()["reasoning_effort"] == "low"
    assert low.cache_suffix() == plain.cache_suffix()  # KB cache paths unaffected
