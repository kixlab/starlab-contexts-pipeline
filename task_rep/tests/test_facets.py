"""
Tests for facet discovery and data structures.

Run with: python -m pytest tests/test_facets.py -v
"""

from types import SimpleNamespace

import pytest

from facets import AnnotationTable, Facet, FacetLabel, VLMFacetExtractor
from facets.prompts.fa_facet_iterative import form_facet_iterative_request
from facets.prompts.formatters import (
    facet_to_message_content,
    facets_to_message_content,
    format_guidelines,
    format_vocabulary,
    schemas_to_message_content,
)
from knowledge_items.schema import (
    KnowledgeBase,
    KnowledgeItem,
    KnowledgeItemType,
    KnowledgeSource,
)
from optimization import AcceptedSchemaSummary, RejectedSchemaSummary
from utils import ModelConfig


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
        user_question="How hard is this technique?",
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
    assert sample_facet.user_question == "How hard is this technique?"
    assert len(sample_facet.vocabulary) == 3
    assert len(sample_facet.guidelines) == 2



def test_facet_valid_scopes():
    """Test all valid scope values."""
    for scope in ["global", "mid-level", "fine-grained"]:
        facet = Facet(
            id="f1", title="Test", title_plural="Tests",
            scope=scope,
            definition="Test", justification="Test",
            user_question="What test value applies?",
            guidelines=[], vocabulary=[],
        )
        assert facet.scope == scope


def test_facet_to_dict(sample_facet):
    """Test Facet serialization."""
    data = sample_facet.to_dict()
    assert data["id"] == "f1"
    assert data["title"] == "Difficulty Level"
    assert data["scope"] == "global"
    assert data["user_question"] == "How hard is this technique?"
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
    assert restored.user_question == sample_facet.user_question
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
        definition="d", justification="j", user_question="What is A?", guidelines=[], vocabulary=[],
    )
    f2 = Facet(
        id="same_id", title="B", title_plural="Bs", scope="mid-level",
        definition="e", justification="k", user_question="What is B?", guidelines=[], vocabulary=[],
    )
    f3 = Facet(
        id="different_id", title="A", title_plural="As", scope="global",
        definition="d", justification="j", user_question="What is A?", guidelines=[], vocabulary=[],
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
    """Test format_guidelines() from facets.prompts.formatters."""
    formatted = format_guidelines(sample_facet.guidelines)
    assert "Label based on required expertise" in formatted
    assert "Consider tool complexity" in formatted


def test_facet_format_vocabulary(sample_facet):
    """Test format_vocabulary() from facets.prompts.formatters."""
    formatted = format_vocabulary(sample_facet.vocabulary)
    assert "beginner" in formatted
    assert "intermediate" in formatted
    assert "advanced" in formatted


def test_facet_format_to_message_content(sample_facet):
    """Test facet_to_message_content()."""
    mc = facet_to_message_content(sample_facet)
    assert len(mc.content) == 1
    assert mc.content[0]["type"] == "text"
    assert "Difficulty Level" in mc.content[0]["text"]
    assert "How hard is this technique?" in mc.content[0]["text"]


def test_facets_to_message_content_includes_label_frequencies(sample_facet):
    kb = KnowledgeBase(items=[
        KnowledgeItem(
            id="i1",
            type=KnowledgeItemType.ATOMIC,
            text="Item 1",
            source=KnowledgeSource(video_id="v1"),
            unit_id="u1",
        ),
        KnowledgeItem(
            id="i2",
            type=KnowledgeItemType.ATOMIC,
            text="Item 2",
            source=KnowledgeSource(video_id="v1"),
            unit_id="u2",
        ),
    ])
    annotations = AnnotationTable({
        ("i1", "f1"): "beginner",
        ("i2", "f1"): "beginner",
    })

    mc = facets_to_message_content([sample_facet], annotations, kb)
    text = mc.content[0]["text"]
    assert "beginner" in text
    assert "<frequency>2</frequency>" in text
    assert "intermediate" in text
    assert "<frequency>0</frequency>" in text


def test_schemas_to_message_content_renders_design_space(sample_facet):
    tool_facet = Facet(
        id="f2",
        title="Tool",
        title_plural="Tools",
        scope="global",
        definition="The tool used for the technique",
        justification="Testing",
        user_question="What tool is being used?",
        guidelines=["Label based on the tool in use"],
        vocabulary=[
            FacetLabel(label="brush", definition="Brush tool"),
            FacetLabel(label="sponge", definition="Sponge tool"),
        ],
    )
    kb = KnowledgeBase(items=[
        KnowledgeItem(
            id="i1",
            type=KnowledgeItemType.ATOMIC,
            text="Item 1",
            source=KnowledgeSource(video_id="v1"),
            unit_id="u1",
        ),
        KnowledgeItem(
            id="i2",
            type=KnowledgeItemType.ATOMIC,
            text="Item 2",
            source=KnowledgeSource(video_id="v1"),
            unit_id="u2",
        ),
    ])
    annotations = AnnotationTable({
        ("i1", "f1"): "beginner",
        ("i2", "f1"): "advanced",
        ("i1", "f2"): "brush",
        ("i2", "f2"): "sponge",
    })
    accepted = [
        AcceptedSchemaSummary(
            id="", facet_ids=[], size=0, D=0.0, eta2=1.0, H=0.0,
        ),
        AcceptedSchemaSummary(
            id="f2", facet_ids=["f2"], size=2, D=0.0, eta2=1.0, H=1.0,
        ),
    ]
    rejected = [RejectedSchemaSummary(
        id="f1|f2",
        facet_ids=["f1", "f2"],
        size=5,
        D=0.0,
        eta2=1.0,
        H=1.0,
        rejected_by_id="f2",
        rejected_by_facet_ids=["f2"],
        MI=1.0,
        refinement_score=1.0,
        refinement_threshold=0.7,
    )]

    accepted.append(AcceptedSchemaSummary(
        id="f1|f2", facet_ids=["f1", "f2"], size=5, D=0.0, eta2=1.0, H=1.0,
        closest_id="f2", closest_facet_ids=["f2"], closest_overlap=0.5,
        member_overlaps={"f1": 0.45, "f2": 0.5},
    ))
    rejected[0] = RejectedSchemaSummary(
        id="f1", facet_ids=["f1"], size=3, D=0.0, eta2=1.0, H=1.0,
        rejected_by_id="f2", rejected_by_facet_ids=["f2"], MI=0.8,
        refinement_score=0.8, refinement_threshold=0.55,
    )
    last = SimpleNamespace(title="Difficulty Level", facet_id="f1",
                           outcome="accepted into the pool", entered=True,
                           previous_schema_ids=["", "f2"])

    mc = schemas_to_message_content(
        accepted, rejected, [sample_facet, tool_facet], annotations, kb,
        refinement_edges=[("f2", "f1|f2")], refinement_threshold=0.55,
        last_submission=last,
    )
    text = mc.content[0]["text"]
    # How it is built, with the run's threshold.
    assert "HOW THE DESIGN SPACE IS BUILT" in text
    assert "overlap above 55% with it" in text and "a single factor or a pair" in text
    # Each factor once, with value frequencies; F IDs follow the factor list.
    assert text.count("<factor id=\"F1\">") == 1
    assert text.count("<factor id=\"F2\">") == 1
    assert "<user_question>What tool is being used?</user_question>" in text
    assert "Design space: kept 1 of 2 single factors and 1 of 1 pairs." in text
    # Kept combinations with why they were kept; the empty schema is not shown.
    assert "- C1: F2 (Tools), 2 cells; kept because no kept combination has fewer values" in text
    assert ("- C2: F1 (Difficulty Levels) + F2 (Tools), 2 cells; kept because F1 alone "
            "carries 45%, F2 alone carries 50%; refines C1") in text
    assert "(no factors)" not in text
    # Pruned singles, and whether they are still used anywhere.
    assert ("- F1 (Difficulty Levels): pruned; C1 (F2 (Tools)) already carries 80% "
            "of its information; still used in C2") in text
    assert "Pruned pairs: 0 of 1." in text
    # What the last submission did.
    assert '- "Difficulty Level" (F1): accepted into the pool.' in text
    assert "- As a single factor: pruned; C1 (F2 (Tools)) already carries 80%" in text
    assert "- In pairs: kept in C2 (with F2 (Tools)); 0 other pairs pruned." in text
    assert "- Elsewhere: nothing else in the design space changed." in text
    # Nothing that scales with the number of segments.
    assert "<signature" not in text
    assert "D=" not in text and "eta2=" not in text


def test_design_space_used_nowhere_and_failed_submission(sample_facet):
    other = Facet(id="f2", title="Tool", title_plural="Tools", scope="global",
                  definition="d", justification="j", user_question="q", guidelines=[],
                  vocabulary=[FacetLabel(label="brush", definition="b")])
    kb = KnowledgeBase(items=[KnowledgeItem(
        id="i1", type=KnowledgeItemType.ATOMIC, text="Item 1",
        source=KnowledgeSource(video_id="v1"), unit_id="u1")])
    accepted = [AcceptedSchemaSummary(id="f1", facet_ids=["f1"], size=3, D=0.0, eta2=1.0, H=1.0)]
    rejected = [
        RejectedSchemaSummary(id="f2", facet_ids=["f2"], size=3, D=0.0, eta2=1.0, H=1.0,
                              rejected_by_id="f1", rejected_by_facet_ids=["f1"], MI=0.9,
                              refinement_score=0.9, refinement_threshold=0.55),
        RejectedSchemaSummary(id="f1|f2", facet_ids=["f1", "f2"], size=6, D=0.0, eta2=1.0,
                              H=1.0, rejected_by_id="f1", rejected_by_facet_ids=["f1"],
                              MI=0.9, refinement_score=0.9, refinement_threshold=0.55),
    ]
    failed = SimpleNamespace(title="Knife", facet_id="x", entered=False,
                             outcome="rejected; independent labelers disagreed too often",
                             previous_schema_ids=["f1"])
    text = schemas_to_message_content(
        accepted, rejected, [sample_facet, other], AnnotationTable(), kb,
        last_submission=failed,
    ).content[0]["text"]
    assert "overlap above 55%" in text  # threshold taken from the rejections
    assert "of its information; used nowhere" in text
    assert ("Pruned pairs: 1 of 1. In 1, one of the pair's own factors already carries "
            "more than 55% of it; in 0, another kept combination does.") in text
    assert ('- "Knife": rejected; independent labelers disagreed too often. '
            "It did not enter the design space.") in text


def test_iterative_prompt_assembles_factor_combinations(sample_facet):
    kb = KnowledgeBase(items=[
        KnowledgeItem(
            id="i1",
            type=KnowledgeItemType.ATOMIC,
            text="Item 1",
            source=KnowledgeSource(video_id="v1"),
            unit_id="u1",
        )
    ])
    annotations = AnnotationTable({("i1", "f1"): "beginner"})
    accepted = [AcceptedSchemaSummary(
        id="f1", facet_ids=["f1"], size=3, D=0.0, eta2=1.0, H=0.0,
    )]

    request = form_facet_iterative_request(
        task="test task",
        facets=[sample_facet],
        components={"schemas"},
        annotations=annotations,
        knowledge_base=kb,
        accepted_schemas=accepted,
        rejected_schemas=[],
    )
    content = request["messages"][1]["content"].content
    text = "\n".join(block["text"] for block in content if block["type"] == "text")
    assert "<frequency>1</frequency>" in text
    assert "Kept combinations:" in text
    assert "<factor id=\"F1\">" in text
    assert "learner's situation" in request["messages"][0]["content"].content[0]["text"]
    assert "Show me the segments for when" in text
    assert "`user_question`" in text


def test_facet_default_metadata():
    """Test that metadata defaults to empty dict."""
    facet = Facet(
        id="f1", title="Test", title_plural="Tests", scope="global",
        definition="Test", justification="Test",
        user_question="What test value applies?",
        guidelines=[], vocabulary=[],
    )
    assert facet.metadata == {}


def test_facet_with_metadata():
    """Test Facet with custom metadata round-trips."""
    facet = Facet(
        id="f1", title="Test", title_plural="Tests", scope="global",
        definition="Test", justification="Test",
        user_question="What test value applies?",
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
    extractor = VLMFacetExtractor(model_config=ModelConfig(category="vlm", provider="openai"))
    assert extractor.model_config.provider == "openai"
    assert extractor.model_config.model == "gpt-4o-mini"
    assert extractor.max_concurrency == 32


def test_vlm_facet_extractor_anthropic():
    """Test VLMFacetExtractor with Anthropic provider."""
    extractor = VLMFacetExtractor(model_config=ModelConfig(category="vlm", provider="anthropic"))
    assert extractor.model_config.provider == "anthropic"
    assert "claude" in extractor.model_config.model


def test_vlm_facet_extractor_custom_model():
    """Test VLMFacetExtractor with custom model."""
    extractor = VLMFacetExtractor(model_config=ModelConfig(category="vlm", provider="openai", model="gpt-4o"))
    assert extractor.model_config.model == "gpt-4o"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
