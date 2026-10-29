"""
Data structures for facets and annotations.

A facet is a piecewise-stable, target-independent, task-relevant dimension
that discriminates knowledge items from one another.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from utils import (
    random_uid,
    MessageContent,
)

FACET_STR_FORMAT = """[F{id}] ({scope}) {title} (Plural: {title_plural}): {definition}
**Guidelines for labeling items**:
{guidelines}
**Vocabulary**:
{vocabulary}
"""

GUIDELINE_STR_FORMAT = """{guideline}"""

LABEL_STR_FORMAT = """[L{id}] `{label}` -- {definition}"""

@dataclass
class FacetLabel:
    """A single label value within a facet vocabulary."""

    label: str
    definition: str
    id: Optional[str] = None

    def __post_init__(self):
        if self.id is None:
            self.id = random_uid()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "definition": self.definition,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> FacetLabel:
        return FacetLabel(
            id=data.get("id"),
            label=data["label"],
            definition=data["definition"],
        )

    def __str__(self) -> str:
        return LABEL_STR_FORMAT.format(
            id=self.label.lower().replace(" ", "_"),
            label=self.label,
            definition=self.definition,
        )

    def format_to_message_content(self) -> MessageContent:
        """Format this facet label as MessageContent for prompt inclusion."""
        return MessageContent(content=[
            {"type": "text", "text": str(self)}
        ])

@dataclass
class Facet:
    """
    A contextual factor that discriminates knowledge items.

    Properties:
    - Piecewise-stable: stays constant for meaningful spans
    - Target-independent: determined by context, not the specific item
    - Task-relevant: affects knowledge applicability

    Attributes:
        id: Unique identifier
        title: Short name (2-3 words)
        title_plural: Plural form
        scope: "global", "mid-level", or "fine-grained"
        definition: What this facet captures
        justification: Why this facet was chosen
        guidelines: Instructions for labeling items
        vocabulary: List of possible labels
    """

    id: str
    title: str
    title_plural: str
    scope: str  # "global" | "mid-level" | "fine-grained"
    definition: str
    justification: str
    guidelines: List[str]
    vocabulary: List[FacetLabel]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.scope not in {"global", "mid-level", "fine-grained"}:
            raise ValueError(f"Invalid scope: {self.scope}")

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, Facet):
            return False
        return self.id == other.id

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "title": self.title,
            "title_plural": self.title_plural,
            "scope": self.scope,
            "definition": self.definition,
            "justification": self.justification,
            "guidelines": self.guidelines,
            "vocabulary": [label.to_dict() for label in self.vocabulary],
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> Facet:
        """Create from dictionary."""
        vocabulary = [FacetLabel.from_dict(v) for v in data["vocabulary"]]
        return Facet(
            id=data["id"],
            title=data["title"],
            title_plural=data["title_plural"],
            scope=data["scope"],
            definition=data["definition"],
            justification=data["justification"],
            guidelines=data["guidelines"],
            vocabulary=vocabulary,
            metadata=data.get("metadata", {}),
        )

    def __str__(self) -> str:
        return FACET_STR_FORMAT.format(
            id=self.id,
            scope=self.scope,
            title=self.title,
            title_plural=self.title_plural,
            definition=self.definition,
            guidelines=self.format_guidelines(),
            vocabulary=self.format_vocabulary(),
        )

    def format_guidelines(self) -> str:
        """Format the guidelines for the facet."""
        if len(self.guidelines) == 0:
            return "No guidelines are available."
        return "\n".join(["\t-" + GUIDELINE_STR_FORMAT.format(guideline=guideline) for guideline in self.guidelines])

    def format_vocabulary(self) -> str:
        """Format the vocabulary for the facet."""
        if len(self.vocabulary) == 0:
            return "No labels are available."
        return "\n".join(["\t" + str(label) for label in self.vocabulary])

    def format_to_message_content(self) -> MessageContent:
        """Format this facet as MessageContent for prompt inclusion."""
        return MessageContent(content=[
            {"type": "text", "text": str(self)}
        ])

    def replace_vocabulary(self, new_vocabulary: List[FacetLabel]):
        """Replace the vocabulary for this facet."""
        self.vocabulary = new_vocabulary


class AnnotationTable:
    """
    Flat mapping of (item_id, facet_id) -> label string.

    Last-write-wins semantics: re-annotating the same (item, facet) pair
    overwrites the previous label (i.e., the most recent annotation is kept).

    Supports merging, filtering by facets or items, and signature computation.
    """

    def __init__(self, data: Optional[Dict[Tuple[str, str], str]] = None):
        self._data: Dict[Tuple[str, str], str] = dict(data) if data else {}

    # ---- dict-like access ----

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, key: Tuple[str, str]) -> str:
        return self._data[key]

    def __setitem__(self, key: Tuple[str, str], value: str):
        self._data[key] = value

    def __contains__(self, key: Tuple[str, str]) -> bool:
        return key in self._data

    def items(self):
        return self._data.items()

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    # ---- merge ----

    def merge(self, other: AnnotationTable) -> AnnotationTable:
        """Return a new table with entries from both; other's entries win on conflict."""
        merged = dict(self._data)
        merged.update(other._data)
        return AnnotationTable(merged)

    def update(self, other: AnnotationTable):
        """In-place merge; other's entries win on conflict."""
        self._data.update(other._data)

    # ---- filter ----

    def filter_facets(self, facet_ids: Iterable[str]) -> AnnotationTable:
        """Return annotations restricted to the given facets."""
        ids = set(facet_ids)
        return AnnotationTable({k: v for k, v in self._data.items() if k[1] in ids})

    def filter_items(self, item_ids: Iterable[str]) -> AnnotationTable:
        """Return annotations restricted to the given items."""
        ids = set(item_ids)
        return AnnotationTable({k: v for k, v in self._data.items() if k[0] in ids})

    # ---- query ----

    def by_facet(self, facet_id: str) -> Dict[str, str]:
        """Return {item_id: label} for a single facet."""
        return {k[0]: v for k, v in self._data.items() if k[1] == facet_id}

    def by_item(self, item_id: str) -> Dict[str, str]:
        """Return {facet_id: label} for a single item."""
        return {k[1]: v for k, v in self._data.items() if k[0] == item_id}

    def signature(
        self,
        item_id: str,
        facets: List[Facet],
        subtype: Optional[str] = None,
    ) -> str:
        """Compute the label signature for an item across the given facets.

        Returns a string like '<subtype:atomic><difficulty:beginner><tool:pandas>'.
        The optional subtype prefix ensures items of different subtypes are
        never grouped into the same cell, even if they share identical facet labels.
        Facets without a label for this item are omitted.
        """
        parts = []
        if subtype is not None:
            parts.append(f"<subtype:{subtype}>")
        for facet in facets:
            key = (item_id, facet.id)
            if key in self._data:
                parts.append(f"<{facet.title}:{self._data[key]}>")
            else:
                print(f"No label found for {key}")
                for k, v in self._data.items():
                    if k[0] == item_id and k[1] == facet.id:
                        print("Found", k, v)
                        break
        return "".join(parts)

    def item_ids(self) -> Set[str]:
        """Return all unique item IDs in this table."""
        return {k[0] for k in self._data}

    def facet_ids(self) -> Set[str]:
        """Return all unique facet IDs in this table."""
        return {k[1] for k in self._data}

    # ---- serialization ----

    def to_list(self) -> List[List[str]]:
        """Serialize to a list of [item_id, facet_id, label] entries."""
        return [[k[0], k[1], v] for k, v in self._data.items()]

    @staticmethod
    def from_list(data: List[List[str]]) -> AnnotationTable:
        """Deserialize from a list of [item_id, facet_id, label] entries."""
        return AnnotationTable({(row[0], row[1]): row[2] for row in data})
