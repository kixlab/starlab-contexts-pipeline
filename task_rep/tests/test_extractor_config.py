"""Extractor settings must not leak between runs or modify the shared preset."""
from knowledge_items.chunk_sampling import FINE_CHUNK_CONFIG
from knowledge_items.extractor import VLMKnowledgeExtractor
from utils import ModelConfig


def test_extractor_chunk_settings_are_independent():
    model = ModelConfig(category="vlm", provider="openai", model="test-model")
    first = VLMKnowledgeExtractor(model_config=model)
    second = VLMKnowledgeExtractor(model_config=model)
    original_size = FINE_CHUNK_CONFIG.size

    first.chunk_config.size = original_size + 3

    assert second.chunk_config.size == original_size
    assert FINE_CHUNK_CONFIG.size == original_size
