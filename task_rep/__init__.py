"""
Task representation layer.

This package defines:
- Core domain models (videos, knowledge items, knowledge base)
- Simple, testable extraction interfaces
- A minimal pipeline to build a knowledge item dataset from a video corpus
- VLM utilities for multimodal extraction
- Dataset preparation (tasks, videos, frames)
"""

from .schema import (  # noqa: F401
    KnowledgeBase,
    KnowledgeItem,
    KnowledgeItemType,
    KnowledgeSource,
    KnowledgeSubtype,
    Video,
    VideoFrame,
    VideoTranscriptSegment,
)

from .pipeline import (  # noqa: F401
    ExtractionConfig,
    run_extraction,
)

from .utils.llm import (  # noqa: F401
    VLMClient,
    MessageContent,
    get_vlm_client,
    llm_enqueue_request,
    llm_get_responses,
    process_api_requests,
)

from .data_prep import (  # noqa: F401
    TaskManager,
    Task,
    VideoDownloader,
    FrameConfig,
    build_dataset,
)
