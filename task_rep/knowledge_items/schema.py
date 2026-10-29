from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

from .taxonomy import AtomicType

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
    transcript: List[VideoTranscriptSegment]
    frames: List[VideoFrame]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def _group_segments(self, segments: List[VideoTranscriptSegment], target_chunks: int) -> List[List[VideoTranscriptSegment]]:
        """
        Group segments to achieve target number of chunks.

        Merges adjacent segments to create approximately target_chunks groups,
        maintaining natural segment boundaries.
        """
        if len(segments) <= target_chunks:
            # Already have fewer segments than target - use each segment as its own group
            return [[seg] for seg in segments]

        # Calculate how many segments per group on average to achieve target_chunks
        segments_per_group = len(segments) / target_chunks

        groups = []
        current_group = []
        current_count = 0

        for i, segment in enumerate(segments):
            current_group.append(segment)
            current_count += 1

            # Check if we should start a new group
            # Either we've accumulated enough segments, or we're at the last segment
            is_last_segment = (i == len(segments) - 1)
            has_enough_segments = current_count >= segments_per_group
            remaining_segments = len(segments) - i - 1
            remaining_groups = target_chunks - len(groups) - 1

            # Start new group if we have enough segments and there are enough remaining
            # to fill the rest of the groups
            if (has_enough_segments and remaining_segments >= remaining_groups) or is_last_segment:
                groups.append(current_group)
                current_group = []
                current_count = 0

                # Stop if we've reached target chunks
                if len(groups) >= target_chunks:
                    break

        return groups

    def get_duration(self) -> float:
        """Get the duration of the video."""
        # transcript_end = self.transcript[-1].end if self.transcript and len(self.transcript) > 0 else 0
        # frames_end = self.frames[-1].time if self.frames and len(self.frames) > 0 else 0
        # return max(transcript_end, frames_end)
        return self.metadata.get("duration", 0)

    def sample_frames(
        self,
        start_time: float,
        end_time: float,
        max_frames: int,
    ) -> List[VideoFrame]:
        """
        Sample frames from a time range according to a chunk size.
        """
        if not self.frames or max_frames < 1:
            return []

        # Filter frames in time range
        relevant_frames = [
            f for f in self.frames
            if start_time <= f.time <= end_time
        ]

        if not relevant_frames:
            return []

        # Sample evenly
        step = len(relevant_frames) / max_frames
        indices = [int((i + 0.5) * step) for i in range(max_frames)]

        return [relevant_frames[i] for i in indices]


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
    # - For STEP items: None (no aggregation)
    # - For TUTORIAL items: None
    subtype: Optional[KnowledgeItemSubtype] = None
    # Unit ID for clustering (assigned by clustering algorithm)
    unit_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain JSON-serializable dict."""
        data = asdict(self)
        # Store enums as their values for easier downstream consumption
        data["type"] = self.type.value
        if self.subtype is not None:
            # For AtomicType enums, use .value; for induced step taxonomy, use string as-is
            data["subtype"] = self.subtype.value if isinstance(self.subtype, AtomicType) else self.subtype
        # unit_id is already serializable (str or None)
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
            unit_id=payload.get("unit_id"),
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "items": [item.to_dict() for item in self.items],
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> KnowledgeBase:
        return KnowledgeBase(items=[KnowledgeItem.from_dict(item) for item in data["items"]])

    def __eq__(self, other: KnowledgeBase) -> bool:
        ### check if the items at the same ids are the same
        item_id_to_item = {item.id: item for item in self.items}
        other_item_id_to_item = {item.id: item for item in other.items}
        if len(item_id_to_item) != len(other_item_id_to_item):
            return False
        for item_id in item_id_to_item.keys():
            if item_id not in other_item_id_to_item:
                return False
            
        for item_id, item in item_id_to_item.items():
            other_item = other_item_id_to_item[item_id]
            item_dict = item.to_dict()
            other_item_dict = other_item.to_dict()
            if item_dict != other_item_dict:
                return False
        return True