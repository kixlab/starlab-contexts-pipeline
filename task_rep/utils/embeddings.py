"""
Embedding API utilities.

Supports text embeddings from various providers with automatic batching.
"""

from __future__ import annotations

import os
from typing import List

import numpy as np
from openai import OpenAI
from google import genai

from .config import ModelConfig

def get_embeddings(
    texts: List[str],
    model_config: ModelConfig,
) -> np.ndarray:
    """
    Get embeddings for a list of texts with automatic batching.

    Args:
        texts: List of text strings to embed
        model: Embedding model name
        provider: API provider (currently only "openai" supported)

    Returns:
        numpy array of embeddings, shape (len(texts), embedding_dim)

    Example:
        >>> texts = ["hello world", "foo bar"]
        >>> embeddings = get_embeddings(texts)
        >>> embeddings.shape
        (2, 1536)
    """
    if not texts:
        return np.array([])

    if model_config.provider == "openai":
        return _get_openai_embeddings(texts, model_config.model)
    elif model_config.provider == "gemini":
        return _get_gemini_embeddings(texts, model_config.model)
    else:
        raise ValueError(f"Unsupported embedding provider: {model_config.provider}")


def _get_gemini_embeddings(texts: List[str], model: str) -> np.ndarray:
    """
    Get embeddings from Gemini API.

    gemini-embedding-2: produces one aggregated embedding per call, so each
    text must be embedded individually.
    gemini-embedding-001: returns one embedding per string in a batch call.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")

    client = genai.Client(api_key=api_key)
    embeddings = []

    def _clean(text: str) -> str:
        t = text.strip() or " "
        return t.replace("\n", " ")

    if "gemini-embedding-2" in model:
        # One call per text — multiple inputs produce a single aggregated embedding
        for text in texts:
            response = client.models.embed_content(model=model, contents=_clean(text))
            embeddings.append(response.embeddings[0].values)
    else:
        # gemini-embedding-001: batch calls return one embedding per input string
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = [_clean(t) for t in texts[i:i + batch_size]]
            response = client.models.embed_content(model=model, contents=batch)
            embeddings.extend([emb.values for emb in response.embeddings])

    return np.array(embeddings)


def _get_openai_embeddings(texts: List[str], model: str) -> np.ndarray:
    """
    Get embeddings from OpenAI API with automatic batching.

    OpenAI limits: 8192 texts per request.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    client = OpenAI(api_key=api_key)

    # Batch processing
    batch_size = 2048
    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        # Clean texts: handle empty strings and newlines
        batch = [text.strip() if text.strip() else " " for text in batch]
        batch = [text.replace("\n", " ") for text in batch]

        response = client.embeddings.create(
            input=batch,
            model=model,
        )
        embeddings.extend([data.embedding for data in response.data])

    return np.array(embeddings)
