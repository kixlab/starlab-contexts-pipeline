"""
Tests for knowledge items extraction and clustering.

Run with: python -m pytest task_rep/tests/test_knowledge_items.py -v
"""

import json
from pathlib import Path

import pytest

from task_rep.knowledge_items import (
    extract_and_cluster_knowledge_items,
    load_knowledge_items,
    save_knowledge_items,
    ClusterConfig,
    KnowledgeItemType,
    KnowledgeItem,
    KnowledgeSource,
    Video,
    VideoTranscriptSegment,
    VideoFrame,
    AtomicType,
)


# Test fixtures
@pytest.fixture
def mock_videos():
    """Create mock videos for testing."""
    return [
        Video(
            id="video1",
            url="https://youtube.com/watch?v=1",
            title="How to Make Sushi",
            original_title="Sushi Tutorial",
            content="Learn to make sushi",
            transcript=[
                VideoTranscriptSegment(text="First, prepare the rice", start=0.0, end=5.0),
                VideoTranscriptSegment(text="Then, cut the fish", start=5.0, end=10.0),
                VideoTranscriptSegment(text="Roll the sushi carefully", start=10.0, end=15.0),
            ],
            frames=[],
        ),
        Video(
            id="video2",
            url="https://youtube.com/watch?v=2",
            title="How to Make Sushi",
            original_title="Sushi Making Guide",
            content="Sushi making guide",
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


# Tests for I/O
def test_save_and_load_knowledge_items(mock_knowledge_items, tmp_path):
    """Test saving and loading knowledge items."""
    output_path = tmp_path / "items.jsonl"

    # Save
    save_knowledge_items(mock_knowledge_items, output_path)
    assert output_path.exists()

    # Load
    loaded_items = load_knowledge_items(output_path)
    assert len(loaded_items) == len(mock_knowledge_items)

    # Check content
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

    # Serialize
    data = item.to_dict()
    assert data["id"] == "test1"
    assert data["type"] == "atomic"
    assert data["subtype"] == "Method - Instruction"
    assert data["unit_id"] == "unit_123"
    assert data["metadata"]["key"] == "value"

    # Deserialize
    restored = KnowledgeItem.from_dict(data)
    assert restored.id == item.id
    assert restored.type == item.type
    assert restored.subtype == item.subtype
    assert restored.unit_id == item.unit_id
    assert restored.metadata == item.metadata


# Tests for clustering
def test_cluster_config():
    """Test ClusterConfig cache suffix generation."""
    config = ClusterConfig(similarity_threshold=0.85, embedding_model="text-embedding-3-small")
    suffix = config.cache_suffix()

    assert "sim085" in suffix
    assert "textembedding" in suffix.lower()


@pytest.mark.skipif(
    not Path("static/datasets/sushi_roll_3.json").exists(),
    reason="sushi_roll_3 dataset not available",
)
def test_extract_and_cluster_with_real_data():
    """
    Integration test with real sushi_roll_3 data.

    Requires:
    - Task sushi_roll_3 to be built: python -m task_rep.scripts.prepare_dataset build sushi_roll_3
    - OPENAI_API_KEY set

    This test:
    1. Loads the sushi_roll_3 dataset
    2. Extracts atomic knowledge items
    3. Clusters them
    4. Verifies caching works
    """
    import os

    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    # Load dataset
    dataset_path = Path("static/datasets/sushi_roll_3.json")
    with open(dataset_path) as f:
        raw = json.load(f)

    videos = []
    for entry in raw:
        videos.append(
            Video(
                id=entry["id"],
                url=entry["url"],
                title=entry["title"],
                original_title=entry.get("original_title", entry["title"]),
                content=entry.get("content", ""),
                transcript=[
                    VideoTranscriptSegment(
                        text=seg["text"],
                        start=float(seg["start"]),
                        end=float(seg["end"]),
                    )
                    for seg in entry.get("transcript", [])
                ],
                frames=[
                    VideoFrame(
                        path=frame["path"],
                        time=float(frame["time"]),
                        caption=frame.get("caption"),
                    )
                    for frame in entry.get("frames", [])
                ],
            )
        )

    assert len(videos) == 3, f"Expected 3 videos, got {len(videos)}"

    # Extract and cluster
    cluster_config = ClusterConfig(similarity_threshold=0.9)
    cache_dir = Path("static/cache_test")

    knowledge_base = extract_and_cluster_knowledge_items(
        task_id="sushi_roll_3_test",
        granularity=KnowledgeItemType.ATOMIC,
        videos=videos,
        base_dir=cache_dir,
        cluster_config=cluster_config,
        use_cache=True,
    )

    # Verify results
    assert len(knowledge_base.items) > 0, "Should extract some knowledge items"

    # Check clustering
    unit_ids = [item.unit_id for item in knowledge_base.items if item.unit_id]
    unique_units = len(set(unit_ids))
    assert unique_units > 0, "Should have at least one cluster"
    assert unique_units < len(knowledge_base.items), "Should cluster similar items"

    # Verify types
    for item in knowledge_base.items:
        assert item.type == KnowledgeItemType.ATOMIC
        assert isinstance(item.subtype, AtomicType) or item.subtype is None

    # Test caching: run again and verify it loads from cache
    knowledge_base2 = extract_and_cluster_knowledge_items(
        task_id="sushi_roll_3_test",
        granularity=KnowledgeItemType.ATOMIC,
        videos=videos,
        base_dir=cache_dir,
        cluster_config=cluster_config,
        use_cache=True,
    )

    assert len(knowledge_base2.items) == len(knowledge_base.items)

    # Cleanup
    import shutil
    if cache_dir.exists():
        shutil.rmtree(cache_dir)


def test_knowledge_base_filtering():
    """Test KnowledgeBase filtering methods."""
    from task_rep.knowledge_items import KnowledgeBase

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
