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

    def fake_gemini(messages, response_format, model=None, temperature=None, seed=None):
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
