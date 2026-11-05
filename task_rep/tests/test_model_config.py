"""ModelConfig: provider defaults, validation, and dict round-trip."""

from __future__ import annotations

import pytest

from utils import ModelConfig


@pytest.mark.parametrize("category, provider, expected", [
    ("vlm", "openai", "gpt-4o-mini"),
    ("vlm", "anthropic", "us.anthropic.claude-3-5-haiku-20241022-v1:0"),
    ("vlm", "gemini", "gemini-2.5-flash"),
    ("embedding", "openai", "text-embedding-3-small"),
    ("embedding", "gemini", "gemini-embedding-2"),
])
def test_default_model_per_provider(category, provider, expected):
    assert ModelConfig(category, provider).model == expected


def test_explicit_model_is_kept():
    assert ModelConfig("vlm", "openai", "custom-model").model == "custom-model"


def test_invalid_category_is_rejected():
    with pytest.raises(ValueError, match="Invalid category"):
        ModelConfig("audio", "openai")


def test_invalid_provider_is_rejected():
    with pytest.raises(ValueError, match="Invalid provider"):
        ModelConfig("vlm", "unknown")


def test_anthropic_embedding_is_not_supported():
    with pytest.raises(ValueError, match="not supported for Anthropic"):
        ModelConfig("embedding", "anthropic")


def test_to_dict_omits_unset_reasoning_effort():
    assert ModelConfig("vlm", "gemini", "x").to_dict() == {
        "category": "vlm", "provider": "gemini", "model": "x"}


def test_dict_round_trip_keeps_reasoning_effort():
    config = ModelConfig("vlm", "openai", "m", reasoning_effort="high")
    data = config.to_dict()
    assert data["reasoning_effort"] == "high"
    assert ModelConfig.from_dict(data) == config


def test_from_dict_fills_default_model_when_missing():
    config = ModelConfig.from_dict({"category": "embedding", "provider": "openai"})
    assert config.model == "text-embedding-3-small"
    assert config.reasoning_effort is None
