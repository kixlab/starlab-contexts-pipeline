from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from .taxonomy import AtomicType, AtomicCategory


class KnowledgeItemType(str, Enum):
    """Types of knowledge items we can extract."""

    ATOMIC = "atomic"
    STEP = "step"
    TUTORIAL = "tutorial"


# KnowledgeItemSubtype can be:
# - AtomicType (from taxonomy) for ATOMIC items
# - str (induced step taxonomy value) for STEP items
# - None for TUTORIAL items
KnowledgeItemSubtype = Union[AtomicType, str]

@dataclass
class VideoTranscriptSegment:
    """Single segment from a video's transcript."""

    text: str
    start: float
    end: float

@dataclass
class VideoFrame:
    """Single frame from a video."""

    path: str
    time: float
    caption: Optional[str] = None

@dataclass
class Video:
    """Source video from which knowledge items are extracted."""

    id: str
    url: str
    title: str
    original_title: str
    content: str
    transcript: List[VideoTranscriptSegment]
    frames: List[VideoFrame]


@dataclass
class KnowledgeSource:
    """
    Origin information for a knowledge item.

    The goal is to always be able to:
    - recover which video and (optionally) which part of that video produced the item
    - reconstruct per-video views for labeling and analysis
    """

    video_id: str
    source_type: str = "video"
    # Optional finer-grained localization within the video
    local_index: Optional[int] = None
    time_range: Optional[Tuple[float, float]] = None
    # Optional hierarchical link (e.g., ATOMIC item pointing to its STEP parent)
    parent_id: Optional[str] = None


@dataclass
class KnowledgeItem:
    """
    Single unit of knowledge

    This is intentionally generic and covers:
    - atomic information (ATOMIC) with taxonomy classification
    - step descriptions (STEP) with induced taxonomy classification
    - tutorial summaries (TUTORIAL)
    """

    id: str
    type: KnowledgeItemType
    text: str
    source: KnowledgeSource
    # Subtype classification:
    # - For ATOMIC items: AtomicType enum (from taxonomy)
    # - For STEP items: str (induced step taxonomy value)
    # - For TUTORIAL items: None
    subtype: Optional[KnowledgeItemSubtype] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain JSON-serializable dict."""
        data = asdict(self)
        # Store enums as their values for easier downstream consumption
        data["type"] = self.type.value
        if self.subtype is not None:
            # For AtomicType enums, use .value; for induced step taxonomy, use string as-is
            data["subtype"] = self.subtype.value if isinstance(self.subtype, AtomicType) else self.subtype
        return data

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> "KnowledgeItem":
        source_data = payload["source"]
        source = KnowledgeSource(
            video_id=source_data["video_id"],
            source_type=source_data.get("source_type", "video"),
            local_index=source_data.get("local_index"),
            time_range=tuple(source_data["time_range"])
            if source_data.get("time_range") is not None
            else None,
            parent_id=source_data.get("parent_id"),
        )
        subtype_val = payload.get("subtype")
        # For ATOMIC items, deserialize as AtomicType enum
        # For STEP items, keep as string (induced step taxonomy value)
        # For TUTORIAL items, will be None
        if subtype_val is not None:
            # Try to deserialize as AtomicType first
            try:
                subtype = AtomicType(subtype_val)
            except ValueError:
                # If not an AtomicType, treat as induced step taxonomy string
                subtype = subtype_val
        else:
            subtype = None

        return KnowledgeItem(
            id=payload["id"],
            type=KnowledgeItemType(payload["type"]),
            text=payload["text"],
            source=source,
            subtype=subtype,
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass
class KnowledgeBase:
    """Lightweight container for a set of knowledge items."""

    items: List[KnowledgeItem]

    def __iter__(self):
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def by_video(
        self,
        types: Optional[Iterable[KnowledgeItemType]] = None,
        subtypes: Optional[Iterable[KnowledgeItemSubtype]] = None,
    ) -> Dict[str, List[KnowledgeItem]]:
        """
        Group items by their source video id, with optional filtering by
        structural type and semantic subtype.
        """
        type_set = set(types) if types is not None else None
        subtype_set = set(subtypes) if subtypes is not None else None

        grouped: Dict[str, List[KnowledgeItem]] = {}
        for item in self.items:
            if type_set is not None and item.type not in type_set:
                continue
            if subtype_set is not None and item.subtype not in subtype_set:
                continue
            grouped.setdefault(item.source.video_id, []).append(item)
        return grouped

    def filter_types(
        self, types: Optional[Iterable[KnowledgeItemType]] = None
    ) -> KnowledgeBase:
        """Return a list of items restricted to the given item types."""
        if types is None:
            return KnowledgeBase(items=list(self.items))
        type_set = set(types)
        return KnowledgeBase(items=[it for it in self.items if it.type in type_set])

    def filter_subtypes(
        self, subtypes: Optional[Iterable[KnowledgeItemSubtype]] = None
    ) -> KnowledgeBase:
        """Return a list of items restricted to the given subtypes."""
        if subtypes is None:
            return KnowledgeBase(items=list(self.items))
        subtype_set = set(subtypes)
        return KnowledgeBase(items=[it for it in self.items if it.subtype in subtype_set])

    def filter_metadata(
        self, metadata: Optional[Dict[str, Any]] = None
    ) -> KnowledgeBase:
        """Return a new knowledge base restricted to the given metadata."""
        if metadata is None:
            return KnowledgeBase(items=list(self.items))
        relevant_items = []
        for item in self.items:
            relevant = True
            for key, value in metadata.items():
                if key in item.metadata and item.metadata[key] != value:
                    relevant = False
                    break
            if relevant:
                relevant_items.append(item)
        return KnowledgeBase(items=relevant_items)
