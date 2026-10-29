"""Utility modules for the task representation framework."""

from .llm import (  # noqa: F401
    VLMClient,
    get_vlm_client,
    llm_enqueue_request,
    llm_get_response,
    llm_get_responses,
    process_api_requests,
)

from .embeddings import (  # noqa: F401
    get_embeddings,
)