"""
Tests for knowledge items extraction, clustering, and data structures.

Run with: python -m pytest tests/test_knowledge_items.py -v
"""

import json
from pathlib import Path

import pytest

from knowledge_items import (
    load_knowledge_items,
    save_knowledge_items,
    load_task_dataset,
    ClusterConfig,
    KnowledgeItemType,
    KnowledgeItem,
    KnowledgeBase,
    KnowledgeSource,
    Video,
    VideoTranscriptSegment,
    VideoFrame,
    AtomicType,
    AtomicCategory,
    get_category,
    get_types_in_category,
)
from knowledge_items.prompts.formatters import video_to_message_content


# ============= Fixtures =============

@pytest.fixture
def mock_videos():
    """Create mock videos for testing."""
    return [
        Video(
            id="video1",
            url="https://youtube.com/watch?v=1",
            title="How to Make Sushi",
            transcript=[
                VideoTranscriptSegment(text="First, prepare the rice", start=0.0, end=5.0),
                VideoTranscriptSegment(text="Then, cut the fish", start=5.0, end=10.0),
                VideoTranscriptSegment(text="Roll the sushi carefully", start=10.0, end=15.0),
            ],
            frames=[
                VideoFrame(path="/tmp/frame1.jpg", time=2.0),
                VideoFrame(path="/tmp/frame2.jpg", time=7.0),
                VideoFrame(path="/tmp/frame3.jpg", time=12.0),
            ],
        ),
        Video(
            id="video2",
            url="https://youtube.com/watch?v=2",
            title="How to Make Sushi",
            transcript=[
                VideoTranscriptSegment(text="Prepare your ingredients", start=0.0, end=5.0),
                VideoTranscriptSegment(text="Season the rice properly", start=5.0, end=10.0),
                VideoTranscriptSegment(text="Cut the fish into slices", start=10.0, end=15.0),
            ],
            frames=[],
        ),
    ]


@pytest.fixture
def mock_knowledge_items():
    """Create mock knowledge items for testing."""
    return [
        KnowledgeItem(
            id="item1",
            type=KnowledgeItemType.ATOMIC,
            text="Prepare the rice by washing it thoroughly",
            source=KnowledgeSource(video_id="video1", local_index=0),
            subtype=AtomicType.METHOD_INSTRUCTION,
        ),
        KnowledgeItem(
            id="item2",
            type=KnowledgeItemType.ATOMIC,
            text="Prepare the rice by rinsing it well",
            source=KnowledgeSource(video_id="video2", local_index=0),
            subtype=AtomicType.METHOD_INSTRUCTION,
        ),
        KnowledgeItem(
            id="item3",
            type=KnowledgeItemType.ATOMIC,
            text="Cut the fish into thin slices",
            source=KnowledgeSource(video_id="video1", local_index=1),
            subtype=AtomicType.METHOD_INSTRUCTION,
        ),
    ]


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create temporary cache directory."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return cache_dir


# ============= I/O Tests =============

def test_save_and_load_knowledge_items(mock_knowledge_items, tmp_path):
    """Test saving and loading knowledge items."""
    output_path = tmp_path / "items.jsonl"

    save_knowledge_items(mock_knowledge_items, output_path)
    assert output_path.exists()

    loaded_items = load_knowledge_items(output_path)
    assert len(loaded_items) == len(mock_knowledge_items)

    for orig, loaded in zip(mock_knowledge_items, loaded_items):
        assert loaded.id == orig.id
        assert loaded.type == orig.type
        assert loaded.text == orig.text
        assert loaded.subtype == orig.subtype


def test_knowledge_item_serialization():
    """Test KnowledgeItem to_dict and from_dict."""
    item = KnowledgeItem(
        id="test1",
        type=KnowledgeItemType.ATOMIC,
        text="Test instruction",
        source=KnowledgeSource(video_id="v1", local_index=0),
        subtype=AtomicType.METHOD_INSTRUCTION,
        unit_id="unit_123",
        metadata={"key": "value"},
    )

    data = item.to_dict()
    assert data["id"] == "test1"
    assert data["type"] == "atomic"
    assert data["subtype"] == "Method - Instruction"
    assert data["unit_id"] == "unit_123"
    assert data["metadata"]["key"] == "value"

    restored = KnowledgeItem.from_dict(data)
    assert restored.id == item.id
    assert restored.type == item.type
    assert restored.subtype == item.subtype
    assert restored.unit_id == item.unit_id
    assert restored.metadata == item.metadata


def test_step_item_serialization():
    """Test serialization of STEP items with string subtype."""
    item = KnowledgeItem(
        id="step1",
        type=KnowledgeItemType.STEP,
        text="Prepare the ingredients",
        source=KnowledgeSource(video_id="v1", local_index=0, time_range=(0.0, 5.0)),
        subtype="Ingredient Preparation",
    )

    data = item.to_dict()
    assert data["type"] == "step"
    assert data["subtype"] == "Ingredient Preparation"
    assert data["source"]["time_range"] == (0.0, 5.0)

    restored = KnowledgeItem.from_dict(data)
    assert restored.type == KnowledgeItemType.STEP
    assert restored.subtype == "Ingredient Preparation"
    assert restored.source.time_range == (0.0, 5.0)


def test_tutorial_item_serialization():
    """Test serialization of TUTORIAL items with None subtype."""
    item = KnowledgeItem(
        id="tutorial1",
        type=KnowledgeItemType.TUTORIAL,
        text="This tutorial covers making sushi from scratch",
        source=KnowledgeSource(video_id="v1", local_index=0),
        subtype=None,
    )

    data = item.to_dict()
    assert data["type"] == "tutorial"
    assert data["subtype"] is None

    restored = KnowledgeItem.from_dict(data)
    assert restored.type == KnowledgeItemType.TUTORIAL
    assert restored.subtype is None


def test_load_knowledge_items_file_not_found(tmp_path):
    """Test that loading from nonexistent path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_knowledge_items(tmp_path / "nonexistent.jsonl")


def test_load_task_dataset(tmp_path):
    """Test loading videos from a dataset JSON file."""
    dataset = [
        {
            "id": "v1",
            "url": "https://youtube.com/watch?v=1",
            "title": "Test Video",
            "transcript": [
                {"text": "Hello world", "start": 0.0, "end": 5.0},
            ],
            "frames": [
                {"path": "/tmp/frame.jpg", "time": 2.0},
            ],
        },
    ]

    path = tmp_path / "test_dataset.json"
    with open(path, "w") as f:
        json.dump(dataset, f)

    videos = load_task_dataset(path)
    assert len(videos) == 1
    assert videos[0].id == "v1"
    assert videos[0].title == "Test Video"
    assert len(videos[0].transcript) == 1
    assert videos[0].transcript[0].text == "Hello world"
    assert len(videos[0].frames) == 1
    assert videos[0].frames[0].time == 2.0


def test_load_task_dataset_file_not_found(tmp_path):
    """Test that loading nonexistent dataset raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_task_dataset(tmp_path / "nonexistent.json")


# ============= KnowledgeItem Formatting Tests =============

def test_knowledge_item_format():
    """Test KnowledgeItem.format() output."""
    item = KnowledgeItem(
        id="item1",
        type=KnowledgeItemType.ATOMIC,
        text="Wash the rice",
        source=KnowledgeSource(video_id="v1"),
        subtype=AtomicType.METHOD_INSTRUCTION,
    )

    formatted = item.format(with_type=True)
    assert "[item1]" in formatted
    assert "Method - Instruction" in formatted
    assert "Wash the rice" in formatted

    formatted_no_type = item.format(with_type=False)
    assert "[item1]" in formatted_no_type
    assert "Method - Instruction" not in formatted_no_type


def test_knowledge_item_format_to_message_content():
    """Test KnowledgeItem.format_to_message_content()."""
    item = KnowledgeItem(
        id="item1",
        type=KnowledgeItemType.ATOMIC,
        text="Wash the rice",
        source=KnowledgeSource(video_id="v1"),
        subtype=AtomicType.METHOD_INSTRUCTION,
    )

    mc = item.format_to_message_content()
    assert len(mc.content) == 1
    assert mc.content[0]["type"] == "text"
    assert "item1" in mc.content[0]["text"]


# ============= Video Tests =============

def test_video_get_duration(mock_videos):
    """Test Video.get_duration()."""
    video = mock_videos[0]
    duration = video.get_duration()
    # Should be max of transcript end (15.0) and last frame time (12.0)
    assert duration == 15.0


def test_video_get_duration_no_transcript():
    """Test get_duration with frames but minimal transcript."""
    video = Video(
        id="v1", url="", title="",
        transcript=[VideoTranscriptSegment(text="hi", start=0.0, end=2.0)],
        frames=[VideoFrame(path="/tmp/f.jpg", time=10.0)],
    )
    assert video.get_duration() == 10.0


def test_video_sample_frames(mock_videos):
    """Test Video.sample_frames()."""
    video = mock_videos[0]

    # Sample 1 frame from 0-15s (has 3 frames)
    sampled = video.sample_frames(start_time=0.0, end_time=15.0, max_frames=1)
    assert len(sampled) == 1

    # Sample 2 frames from 0-15s
    sampled = video.sample_frames(start_time=0.0, end_time=15.0, max_frames=2)
    assert len(sampled) == 2

    # Sample from narrow range (1 frame in range, but max_frames=5 produces repeats)
    sampled = video.sample_frames(start_time=6.0, end_time=8.0, max_frames=1)
    assert len(sampled) == 1  # Only frame at 7.0 is in range

    # No frames in range
    sampled = video.sample_frames(start_time=20.0, end_time=30.0, max_frames=5)
    assert len(sampled) == 0


def test_video_sample_frames_no_frames():
    """Test sample_frames when video has no frames."""
    video = Video(
        id="v1", url="", title="",
        transcript=[VideoTranscriptSegment(text="hi", start=0.0, end=5.0)],
        frames=[],
    )
    assert video.sample_frames(0.0, 5.0, 3) == []


def test_video_format_to_message_content_text_only(mock_videos):
    """Test video_to_message_content without vision."""
    video = mock_videos[1]  # No frames
    mc = video_to_message_content(video, use_vision=False, frames_per_group=1)
    assert len(mc.content) > 0
    # All content should be text blocks
    for block in mc.content:
        assert block["type"] == "text"


def test_video_format_to_message_content_no_transcript():
    """Test that video_to_message_content raises on empty transcript."""
    video = Video(
        id="v1", url="", title="",
        transcript=[],
        frames=[VideoFrame(path="/tmp/f.jpg", time=1.0)],
    )
    with pytest.raises(ValueError, match="must have transcript"):
        video_to_message_content(video, use_vision=False, frames_per_group=1)


def test_video_format_to_message_content_vision_no_frames():
    """Test that video_to_message_content raises when vision=True but no frames."""
    video = Video(
        id="v1", url="", title="",
        transcript=[VideoTranscriptSegment(text="hi", start=0.0, end=5.0)],
        frames=[],
    )
    with pytest.raises(ValueError, match="vision is enabled"):
        video_to_message_content(video, use_vision=True, frames_per_group=1)


def test_video_group_segments(mock_videos):
    """Test Video._group_segments for chunking."""
    video = mock_videos[0]

    # Target 1 chunk -> all segments in one group
    groups = video._group_segments(video.transcript, target_chunks=1)
    assert len(groups) == 1
    assert len(groups[0]) == 3

    # Target 3 chunks -> one segment per group
    groups = video._group_segments(video.transcript, target_chunks=3)
    assert len(groups) == 3

    # Target more chunks than segments -> one segment per group
    groups = video._group_segments(video.transcript, target_chunks=10)
    assert len(groups) == 3


# ============= Taxonomy Tests =============

def test_get_category():
    """Test mapping from AtomicType to AtomicCategory."""
    assert get_category(AtomicType.METHOD_INSTRUCTION) == AtomicCategory.METHOD
    assert get_category(AtomicType.SUPPLEMENTARY_TIP) == AtomicCategory.SUPPLEMENTARY
    assert get_category(AtomicType.EXPLANATION_JUSTIFICATION) == AtomicCategory.EXPLANATION
    assert get_category(AtomicType.DESCRIPTION_STATUS) == AtomicCategory.DESCRIPTION
    assert get_category(AtomicType.CONCLUSION_OUTCOME) == AtomicCategory.CONCLUSION


def test_get_types_in_category():
    """Test getting all types in a category."""
    method_types = get_types_in_category(AtomicCategory.METHOD)
    assert AtomicType.METHOD_INSTRUCTION in method_types
    assert AtomicType.METHOD_SUBGOAL in method_types
    assert AtomicType.METHOD_TOOL in method_types
    assert len(method_types) == 3


# ============= KnowledgeBase Tests =============

def test_knowledge_base_filtering():
    """Test KnowledgeBase filtering methods."""
    items = [
        KnowledgeItem(
            id="1",
            type=KnowledgeItemType.ATOMIC,
            text="Instruction 1",
            source=KnowledgeSource(video_id="v1"),
            subtype=AtomicType.METHOD_INSTRUCTION,
        ),
        KnowledgeItem(
            id="2",
            type=KnowledgeItemType.ATOMIC,
            text="Tip 1",
            source=KnowledgeSource(video_id="v1"),
            subtype=AtomicType.SUPPLEMENTARY_TIP,
        ),
        KnowledgeItem(
            id="3",
            type=KnowledgeItemType.STEP,
            text="Step 1",
            source=KnowledgeSource(video_id="v2"),
            subtype="Prepare ingredients",
        ),
    ]

    kb = KnowledgeBase(items=items)

    # Filter by type
    atomic_only = kb.filter_types([KnowledgeItemType.ATOMIC])
    assert len(atomic_only.items) == 2

    # Filter by subtype
    instructions_only = kb.filter_subtypes([AtomicType.METHOD_INSTRUCTION])
    assert len(instructions_only.items) == 1
    assert instructions_only.items[0].id == "1"

    # Group by video
    by_video = kb.by_video()
    assert len(by_video["v1"]) == 2
    assert len(by_video["v2"]) == 1


def test_knowledge_base_by_video_with_filters():
    """Test KnowledgeBase.by_video() with type and subtype filters."""
    items = [
        KnowledgeItem(
            id="1", type=KnowledgeItemType.ATOMIC, text="Instruction",
            source=KnowledgeSource(video_id="v1"),
            subtype=AtomicType.METHOD_INSTRUCTION,
        ),
        KnowledgeItem(
            id="2", type=KnowledgeItemType.ATOMIC, text="Tip",
            source=KnowledgeSource(video_id="v1"),
            subtype=AtomicType.SUPPLEMENTARY_TIP,
        ),
        KnowledgeItem(
            id="3", type=KnowledgeItemType.STEP, text="Step",
            source=KnowledgeSource(video_id="v1"),
            subtype="Prepare ingredients",
        ),
    ]

    kb = KnowledgeBase(items=items)

    # Filter by type
    by_video = kb.by_video(types=[KnowledgeItemType.ATOMIC])
    assert len(by_video["v1"]) == 2

    # Filter by subtype
    by_video = kb.by_video(subtypes=[AtomicType.METHOD_INSTRUCTION])
    assert len(by_video["v1"]) == 1

    # Filter by both
    by_video = kb.by_video(
        types=[KnowledgeItemType.ATOMIC],
        subtypes=[AtomicType.METHOD_INSTRUCTION],
    )
    assert len(by_video["v1"]) == 1
    assert by_video["v1"][0].id == "1"


def test_knowledge_base_filter_metadata():
    """Test KnowledgeBase.filter_metadata()."""
    items = [
        KnowledgeItem(
            id="1", type=KnowledgeItemType.ATOMIC, text="Item 1",
            source=KnowledgeSource(video_id="v1"),
            metadata={"confidence": "high", "source": "manual"},
        ),
        KnowledgeItem(
            id="2", type=KnowledgeItemType.ATOMIC, text="Item 2",
            source=KnowledgeSource(video_id="v1"),
            metadata={"confidence": "low"},
        ),
        KnowledgeItem(
            id="3", type=KnowledgeItemType.ATOMIC, text="Item 3",
            source=KnowledgeSource(video_id="v1"),
            metadata={},
        ),
    ]

    kb = KnowledgeBase(items=items)

    # filter_metadata only excludes items where key exists AND value mismatches
    # Items missing the key pass through
    filtered = kb.filter_metadata({"confidence": "high"})
    assert len(filtered.items) == 2  # Item 1 matches, Item 3 has no key (passes)
    assert filtered.items[0].id == "1"
    assert filtered.items[1].id == "3"

    # Filter that only Item 1 has both keys for
    filtered = kb.filter_metadata({"confidence": "high", "source": "manual"})
    assert len(filtered.items) == 2  # Item 1 matches both, Item 3 has neither key

    # None filter returns all
    filtered = kb.filter_metadata(None)
    assert len(filtered.items) == 3


def test_knowledge_base_iter_and_len():
    """Test KnowledgeBase __iter__ and __len__."""
    items = [
        KnowledgeItem(
            id="1", type=KnowledgeItemType.ATOMIC, text="Item 1",
            source=KnowledgeSource(video_id="v1"),
        ),
        KnowledgeItem(
            id="2", type=KnowledgeItemType.ATOMIC, text="Item 2",
            source=KnowledgeSource(video_id="v1"),
        ),
    ]
    kb = KnowledgeBase(items=items)

    assert len(kb) == 2
    collected = [item for item in kb]
    assert len(collected) == 2
    assert collected[0].id == "1"


# ============= Clustering Config Tests =============

def test_cluster_config():
    """Test ClusterConfig cache suffix generation."""
    config = ClusterConfig(similarity_threshold=0.85, embedding_model="text-embedding-3-small")
    suffix = config.cache_suffix()

    assert "sim085" in suffix
    assert "textembedding" in suffix.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
