"""
ItemScopeConfig: bundles KnowledgeItemType with an atomic subtype preset.

Lives in a separate module to avoid circular imports:
  schema.py  → taxonomy.py  (AtomicType)
  item_scope.py → taxonomy.py + schema.py  (KnowledgeItemType)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from .schema import KnowledgeItemType
from .taxonomy import (
    AtomicSubtypeConfig,
    ATOMIC_ALL,
    ATOMIC_SUBTYPE_PRESETS,
)


@dataclass
class ItemScopeConfig:
    """
    Defines which items the algorithm optimizes over.

    Bundles ``item_type`` (ATOMIC / STEP / TUTORIAL) with an
    ``atomic_subtype`` preset so the combined scope is expressible as a
    single, human-readable cache token.

    For non-ATOMIC item types the ``atomic_subtype`` field is ignored.
    """

    item_type: KnowledgeItemType
    atomic_subtype: AtomicSubtypeConfig = field(default_factory=lambda: ATOMIC_ALL)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_type": self.item_type.value,
            "atomic_subtype": self.atomic_subtype.name,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ItemScopeConfig":
        preset_name = data.get("atomic_subtype", "all")
        return ItemScopeConfig(
            item_type=KnowledgeItemType(data["item_type"]),
            atomic_subtype=ATOMIC_SUBTYPE_PRESETS.get(preset_name, ATOMIC_ALL),
        )
