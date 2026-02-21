"""
Tests for facet discovery and data structures.

Run with: python -m pytest tests/test_facets.py -v
"""

import pytest

from facets import Facet, FacetLabel, VLMFacetExtractor


# ============= FacetLabel Tests =============

def test_facet_label_creation():
    """Test FacetLabel creation with auto-generated id."""
    label = FacetLabel(label="beginner", definition="For newcomers")
    assert label.label == "beginner"
    assert label.definition == "For newcomers"
    assert label.id is not None  # Auto-generated


def test_facet_label_explicit_id():
    """Test FacetLabel creation with explicit id."""
    label = FacetLabel(label="beginner", definition="For newcomers", id="custom_id")
    assert label.id == "custom_id"


def test_facet_label_to_dict():
    """Test FacetLabel serialization."""
    label = FacetLabel(label="beginner", definition="For newcomers", id="l1")
    data = label.to_dict()
    assert data["id"] == "l1"
    assert data["label"] == "beginner"
    assert data["definition"] == "For newcomers"


def test_facet_label_from_dict():
    """Test FacetLabel deserialization."""
    data = {"id": "l1", "label": "beginner", "definition": "For newcomers"}
    label = FacetLabel.from_dict(data)
    assert label.id == "l1"
    assert label.label == "beginner"
    assert label.definition == "For newcomers"


def test_facet_label_roundtrip():
    """Test FacetLabel serialization round-trip."""
    original = FacetLabel(label="advanced", definition="For experts", id="l2")
    restored = FacetLabel.from_dict(original.to_dict())
    assert restored.id == original.id
    assert restored.label == original.label
    assert restored.definition == original.definition


def test_facet_label_str():
    """Test FacetLabel string formatting."""
    label = FacetLabel(label="Beginner Level", definition="For newcomers", id="l1")
    s = str(label)
    assert "Beginner Level" in s
    assert "For newcomers" in s


def test_facet_label_format_to_message_content():
    """Test FacetLabel.format_to_message_content()."""
    label = FacetLabel(label="beginner", definition="For newcomers", id="l1")
    mc = label.format_to_message_content()
    assert len(mc.content) == 1
    assert mc.content[0]["type"] == "text"
    assert "beginner" in mc.content[0]["text"]


# ============= Facet Tests =============

@pytest.fixture
def sample_facet():
    """Create a sample facet for testing."""
    return Facet(
        id="f1",
        title="Difficulty Level",
        title_plural="Difficulty Levels",
        scope="global",
        definition="The skill level required for this technique",
        justification="Different techniques require different skill levels",
        guidelines=["Label based on required expertise", "Consider tool complexity"],
        vocabulary=[
            FacetLabel(label="beginner", definition="Basic techniques", id="l1"),
            FacetLabel(label="intermediate", definition="Moderate complexity", id="l2"),
            FacetLabel(label="advanced", definition="Expert-level", id="l3"),
        ],
    )


def test_facet_creation(sample_facet):
    """Test Facet creation with valid scope."""
    assert sample_facet.id == "f1"
    assert sample_facet.title == "Difficulty Level"
    assert sample_facet.scope == "global"
    assert len(sample_facet.vocabulary) == 3
    assert len(sample_facet.guidelines) == 2


def test_facet_invalid_scope():
    """Test that invalid scope raises ValueError."""
    with pytest.raises(ValueError, match="Invalid scope"):
        Facet(
            id="f1", title="Test", title_plural="Tests",
            scope="invalid",
            definition="Test", justification="Test",
            guidelines=[], vocabulary=[],
        )


def test_facet_valid_scopes():
    """Test all valid scope values."""
    for scope in ["global", "mid-level", "fine-grained"]:
        facet = Facet(
            id="f1", title="Test", title_plural="Tests",
            scope=scope,
            definition="Test", justification="Test",
            guidelines=[], vocabulary=[],
        )
        assert facet.scope == scope


def test_facet_to_dict(sample_facet):
    """Test Facet serialization."""
    data = sample_facet.to_dict()
    assert data["id"] == "f1"
    assert data["title"] == "Difficulty Level"
    assert data["scope"] == "global"
    assert len(data["vocabulary"]) == 3
    assert data["vocabulary"][0]["label"] == "beginner"


def test_facet_from_dict(sample_facet):
    """Test Facet deserialization."""
    data = sample_facet.to_dict()
    restored = Facet.from_dict(data)
    assert restored.id == sample_facet.id
    assert restored.title == sample_facet.title
    assert restored.scope == sample_facet.scope
    assert restored.definition == sample_facet.definition
    assert restored.justification == sample_facet.justification
    assert restored.guidelines == sample_facet.guidelines
    assert len(restored.vocabulary) == len(sample_facet.vocabulary)


def test_facet_roundtrip(sample_facet):
    """Test Facet serialization round-trip preserves all fields."""
    data = sample_facet.to_dict()
    restored = Facet.from_dict(data)

    for orig_label, restored_label in zip(sample_facet.vocabulary, restored.vocabulary):
        assert orig_label.id == restored_label.id
        assert orig_label.label == restored_label.label
        assert orig_label.definition == restored_label.definition


def test_facet_hash_and_eq():
    """Test Facet __hash__ and __eq__ are based on id."""
    f1 = Facet(
        id="same_id", title="A", title_plural="As", scope="global",
        definition="d", justification="j", guidelines=[], vocabulary=[],
    )
    f2 = Facet(
        id="same_id", title="B", title_plural="Bs", scope="mid-level",
        definition="e", justification="k", guidelines=[], vocabulary=[],
    )
    f3 = Facet(
        id="different_id", title="A", title_plural="As", scope="global",
        definition="d", justification="j", guidelines=[], vocabulary=[],
    )

    assert f1 == f2  # Same id
    assert f1 != f3  # Different id
    assert hash(f1) == hash(f2)

    # Can be used in sets
    facet_set = {f1, f2, f3}
    assert len(facet_set) == 2


def test_facet_str(sample_facet):
    """Test Facet string formatting."""
    s = str(sample_facet)
    assert "Difficulty Level" in s
    assert "global" in s
    assert "beginner" in s
    assert "Guidelines" in s or "guidelines" in s.lower()


def test_facet_format_guidelines(sample_facet):
    """Test Facet.format_guidelines()."""
    formatted = sample_facet.format_guidelines()
    assert "Label based on required expertise" in formatted
    assert "Consider tool complexity" in formatted


def test_facet_format_vocabulary(sample_facet):
    """Test Facet.format_vocabulary()."""
    formatted = sample_facet.format_vocabulary()
    assert "beginner" in formatted
    assert "intermediate" in formatted
    assert "advanced" in formatted


def test_facet_format_to_message_content(sample_facet):
    """Test Facet.format_to_message_content()."""
    mc = sample_facet.format_to_message_content()
    assert len(mc.content) == 1
    assert mc.content[0]["type"] == "text"
    assert "Difficulty Level" in mc.content[0]["text"]


def test_facet_default_metadata():
    """Test that metadata defaults to empty dict."""
    facet = Facet(
        id="f1", title="Test", title_plural="Tests", scope="global",
        definition="Test", justification="Test",
        guidelines=[], vocabulary=[],
    )
    assert facet.metadata == {}


def test_facet_with_metadata():
    """Test Facet with custom metadata round-trips."""
    facet = Facet(
        id="f1", title="Test", title_plural="Tests", scope="global",
        definition="Test", justification="Test",
        guidelines=[], vocabulary=[],
        metadata={"iteration": 3, "source": "discovery"},
    )
    data = facet.to_dict()
    assert data["metadata"]["iteration"] == 3

    restored = Facet.from_dict(data)
    assert restored.metadata["iteration"] == 3
    assert restored.metadata["source"] == "discovery"


# ============= VLMFacetExtractor Tests =============

def test_vlm_facet_extractor_defaults():
    """Test VLMFacetExtractor default initialization."""
    extractor = VLMFacetExtractor(task="sushi")
    assert extractor.task == "sushi"
    assert extractor.provider == "openai"
    assert extractor.model == "gpt-4o-mini"
    assert extractor.max_concurrency == 32


def test_vlm_facet_extractor_anthropic():
    """Test VLMFacetExtractor with Anthropic provider."""
    extractor = VLMFacetExtractor(task="sushi", provider="anthropic")
    assert extractor.provider == "anthropic"
    assert "claude" in extractor.model


def test_vlm_facet_extractor_custom_model():
    """Test VLMFacetExtractor with custom model."""
    extractor = VLMFacetExtractor(task="sushi", provider="openai", model="gpt-4o")
    assert extractor.model == "gpt-4o"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
